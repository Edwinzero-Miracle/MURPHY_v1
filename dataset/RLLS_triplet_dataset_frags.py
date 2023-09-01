
import json
import os
import argparse
import pickle
from collections import defaultdict

import cv2
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm


SUBPATH_TO_LABEL = '/FinalLabels/'
SUBPATH_TO_IMAGE = '/Images/'
TRIPLET_DICT_JSON = 'RLLS12M_annotation_dict.json'


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def read_json(filename):
    with open(filename, encoding='utf-8', errors="ignore") as para_json_file:
        data = json.load(para_json_file)
    return data


def read_pkl_trial(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    return data


class RLLSVideoSeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx  # all frame idx with offset from videos
        # [ (0,1,2,3) (4,5,6,7) (8,9,10,11) (12,13,14,15) ... ] if len=4

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


# avoid sampling over range
def get_useful_start_idx(sequence_length: int, list_each_length: list, stride: int):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)

        count += list_each_length[i]
    return idx


class RLLSVideoDataset(Dataset):
    def __init__(self, data_path: str, trial_list: list, sequence_frag_len: int,
                 default_fps: int = 5, max_num_frames: int = 0,
                 shuffle_in_batches: bool = False, sort_in_batches: bool = False,
                 crop_img_ui: bool = False,
                 transform: transforms = None,
                 loader=pil_loader):
        """
        The key of indexing is the d,k,i key in those dictionaries
        use idx_to_dictionary for __get_item__()
        the value of idx_to_dictionary returns d, k, i
        then d, k can fetch camera_parameters, and d, k, i can fetch the frame name.
        In addition, d, k, i can index the transcript for corresponding epe, d1, and mae ...

        :param data_path:  The root path of the RLLS_dataset
        :param trial_list: e.g. ['AA0', 'AA1', 'AA2', 'AA3' ...]
        :param max_num_frames: toy, debug, deploy can be set to different values,
        :param loader:
        """
        super(RLLSVideoDataset, self).__init__()
        # the below variables are for batching
        self.num_train_we_use = 0
        self.sequence_frag_len = 0
        self.frag_stride = 0
        self.total_frames = 0

        self.trial_json_filenames = None
        self.LABEL_STEP_TO_CAT = {'Idle': 0}
        self.LABEL_TASK_TO_CAT = {'Idle': 0}
        self.LABEL_AA_TO_CAT = {'Idle': 0}
        self.LABEL_INSTRUMENT_TO_CAT = {'Idle': 0}
        self.LABEL_ACT_TO_CAT = {'Idle': 0}
        self.LABEL_OBJECT_TO_CAT = {'Idle': 0}
        self.LABEL_IAO_TO_CAT = {}
        self.LABEL_IAO_TO_AA = {}
        self.LABEL_IAO_CAT_TO_AA_CAT = {}

        self.data_buffer = defaultdict(list)
        self.max_num_frames = max_num_frames
        self.sequence_frag_len = sequence_frag_len
        self.shuffle_in_batches = shuffle_in_batches
        self.sort_in_batches = sort_in_batches

        self.base_path = data_path
        self.json_dict_filename = self.base_path + TRIPLET_DICT_JSON
        self.load_label_dict(self.json_dict_filename)
        print("   [RLLSDataset :: Video frag Triplet]:: load dict file from: ", self.json_dict_filename, "  (True)")

        self.fps = default_fps
        self.trial_list = trial_list
        self.load_trial_data(self.trial_list, self.fps)

        self.loader = loader
        self.crop_img_ui = crop_img_ui

        self.transform = transform

        print('[RLLSDataset :: Video frag Triplet] num of properties  : {:6d} = 1 img, 6 labels'.format(
            len(self.data_buffer)))
        print('[RLLSDataset :: Video frag Triplet] num of frames  : {:6d}'.format(len(self.data_buffer['img_name'])))
        print('==> [RLLS Dataset Video frag] :: Load data successfully!, including: ', trial_list)

    def load_label_dict(self, json_dict_filename):
        if json_dict_filename is None:
            raise FileNotFoundError

        raw_label_dict = read_json(json_dict_filename)
        # print(raw_label_dict)

        count = 1
        self.LABEL_INSTRUMENT_TO_CAT = {'Idle': 0}
        for ins_label in raw_label_dict['instrument']['label']:
            self.LABEL_INSTRUMENT_TO_CAT[ins_label] = count
            count += 1
        count = 1
        self.LABEL_ACT_TO_CAT = {'Idle': 0}
        for act_label in raw_label_dict['action']['label']:
            self.LABEL_ACT_TO_CAT[act_label] = count
            count += 1
        count = 1
        self.LABEL_OBJECT_TO_CAT = {'Idle': 0}
        for obj_label in raw_label_dict['object']['label']:
            self.LABEL_OBJECT_TO_CAT[obj_label] = count
            count += 1
        count = 1
        self.LABEL_STEP_TO_CAT = {'Idle': 0}
        for step_label in raw_label_dict['step']['label']:
            self.LABEL_STEP_TO_CAT[step_label] = count
            count += 1
        count = 1
        self.LABEL_TASK_TO_CAT = {'Idle': 0}
        for task_label in raw_label_dict['task']['label']:
            self.LABEL_TASK_TO_CAT[task_label] = count
            count += 1

        self.LABEL_AA_TO_CAT = {'Idle': 0}
        for i in range(0, len(raw_label_dict['triplet']['label'])):
            self.LABEL_AA_TO_CAT[raw_label_dict['triplet']['label'][i]] = raw_label_dict['triplet']['id'][i]

        count = 1
        self.LABEL_IAO_TO_AA = {}
        self.LABEL_IAO_TO_CAT = {}
        for aa, iao in raw_label_dict['triplet']['tuple'].items():
            self.LABEL_IAO_TO_AA[iao[0], iao[1], iao[2]] = aa
            self.LABEL_IAO_TO_CAT[iao[0], iao[1], iao[2]] = self.LABEL_AA_TO_CAT[aa]
            self.LABEL_IAO_CAT_TO_AA_CAT[self.LABEL_INSTRUMENT_TO_CAT[iao[0]],
                                         self.LABEL_ACT_TO_CAT[iao[1]], self.LABEL_OBJECT_TO_CAT[iao[2]]] = \
            self.LABEL_AA_TO_CAT[aa]
            count += 1

        print("[RLLS_triplet_dataset_trialwise.py] :: note :: Successfully load all surgical label dictionary ! ")

    def load_trial_data(self, trial_list: list, fps: int):
        self.trial_json_filenames = []
        for subject in trial_list:
            filename = self.base_path + SUBPATH_TO_LABEL + subject + '_fps' + str(fps) + '.json'
            if not os.path.exists(filename):
                print("Trial json file not exist: ", filename)
            self.trial_json_filenames.append(filename)
        print("total load json label trials: ", len(self.trial_json_filenames))

        self.offset = []
        self.frames_in_trial = []
        self.num_of_videos = 0
        # note: key =[image filename] ; value = [step, task, triplet, instrument, action, object]
        tmp_data = defaultdict(list)
        # iterate all videos
        for i, filename in enumerate(self.trial_json_filenames):
            label_data = read_json(filename)

            # prepare values for indexing
            self.num_of_videos = self.num_of_videos + 1
            if len(self.offset) == 0:
                self.offset.append(0)
                self.frames_in_trial.append(len(label_data['step'].values()))
            else:
                self.offset.append(self.frames_in_trial[i - 1] + self.offset[i - 1])
                self.frames_in_trial.append(len(label_data['step'].values()))

            count = 1
            for img_name, step in label_data['step'].items():
                img_name_key = self.trial_list[i] + '/' + img_name
                # print(img_name_key, ", ", step, ", ", self.LABEL_STEP_TO_CAT[step])
                tmp_data[img_name_key].append(self.LABEL_STEP_TO_CAT[step])
                if count >= self.max_num_frames:
                    break
                count += 1

            count = 1
            for img_name, task in label_data['task'].items():
                img_name_key = self.trial_list[i] + '/' + img_name
                tmp_data[img_name_key].append(self.LABEL_TASK_TO_CAT[task])
                if count >= self.max_num_frames:
                    break
                count += 1

            count = 1
            for img_name, triplet in label_data['triplet'].items():
                img_name_key = self.trial_list[i] + '/' + img_name
                tmp_data[img_name_key].append(self.LABEL_AA_TO_CAT[triplet])
                if count >= self.max_num_frames:
                    break
                count += 1

            count = 1
            for img_name, instrument in label_data['instrument'].items():
                img_name_key = self.trial_list[i] + '/' + img_name
                tmp_data[img_name_key].append(instrument)
                if count >= self.max_num_frames:
                    break
                count += 1

            count = 1
            for img_name, action in label_data['action'].items():
                img_name_key = self.trial_list[i] + '/' + img_name
                tmp_data[img_name_key].append(action)
                if count >= self.max_num_frames:
                    break
                count += 1

            count = 1
            for img_name, object in label_data['object'].items():
                img_name_key = self.trial_list[i] + '/' + img_name
                tmp_data[img_name_key].append(object)
                if count >= self.max_num_frames:
                    break
                count += 1

        self.total_frames = sum(self.frames_in_trial)
        self.data_buffer = defaultdict(list)
        if self.max_num_frames <= 0:
            self.max_num_frames = len(tmp_data.items())

        for img_name, tmp_labels in tmp_data.items():
            self.data_buffer['img_name'].append(self.base_path + SUBPATH_TO_IMAGE + str(self.fps) + 'fps/' + img_name)
            self.data_buffer['step'].append(tmp_labels[0])
            self.data_buffer['task'].append(tmp_labels[1])
            self.data_buffer['triplet'].append(tmp_labels[2])
            self.data_buffer['instrument'].append(tmp_labels[3])
            self.data_buffer['action'].append(tmp_labels[4])
            self.data_buffer['object'].append(tmp_labels[5])

        del tmp_data

    def _filter_idle_frames(self):
        tmp_data_buffer = {'img_name': [],
                           'step': [],
                           'task': [],
                           'triplet': [],
                           'instrument': [],
                           'action': [],
                           'object': []}
        for i in range(0, len(self.data_buffer['img_name'])):
            if self.data_buffer['step'][i] == 0 or \
                    self.data_buffer['step'][i] == 0 or \
                    self.data_buffer['step'][i] == 0 or \
                    self.data_buffer['step'][i] == 0 or \
                    self.data_buffer['step'][i] == 0 or \
                    self.data_buffer['step'][i] == 0:
                continue
            tmp_data_buffer['img_name'].append(self.data_buffer['img_name'][i])
            tmp_data_buffer['step'].append(self.data_buffer['step'][i])
            tmp_data_buffer['task'].append(self.data_buffer['task'][i])
            tmp_data_buffer['triplet'].append(self.data_buffer['triplet'][i])
            tmp_data_buffer['instrument'].append(self.data_buffer['instrument'][i])
            tmp_data_buffer['action'].append(self.data_buffer['action'][i])
            tmp_data_buffer['object'].append(self.data_buffer['object'][i])

        self.data_buffer['img_name'] = tmp_data_buffer['img_name'].copy()
        self.data_buffer['step'] = tmp_data_buffer['step'].copy()
        self.data_buffer['task'] = tmp_data_buffer['task'].copy()
        self.data_buffer['triplet'] = tmp_data_buffer['triplet'].copy()
        self.data_buffer['instrument'] = tmp_data_buffer['instrument'].copy()
        self.data_buffer['action'] = tmp_data_buffer['action'].copy()
        self.data_buffer['object'] = tmp_data_buffer['object'].copy()
        del tmp_data_buffer
        return

    def process_image(self, img_name):
        img = self.loader(img_name)
        if self.crop_img_ui:
            img = img.crop((90, 72, 1190, 952))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img)
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        return img

    # Get a frame
    def __getitem__(self, index):
        res = {}
        img_name = self.data_buffer['img_name'][index]
        res['img'] = self.loader(img_name)
        if self.crop_img_ui:
            res['img'] = res['img'].crop((90, 72, 1190, 952))

        if self.transform is not None:
            res['img'] = self.transform(res['img'])
        else:
            res['img'] = np.array(res['img'])
            res['img'] = cv2.resize(res['img'], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

        res['step'] = self.data_buffer['step'][index]
        res['task'] = self.data_buffer['task'][index]
        res['triplet'] = self.data_buffer['triplet'][index]
        res['instrument'] = self.data_buffer['instrument'][index]
        res['action'] = self.data_buffer['action'][index]
        res['object'] = self.data_buffer['object'][index]
        res['img_name'] = self.data_buffer['img_name'][index]  # add (11.22.2022)

        return res

    def __len__(self):
        return len(self.data_buffer['img_name'])

    def prepare(self, sequence_frag_len: int, frag_stride: int):
        self.sequence_frag_len = sequence_frag_len
        self.frag_stride = frag_stride
        assert frag_stride <= sequence_frag_len
        assert len(self.frames_in_trial) == self.num_of_videos
        offset_count = 0
        valid_start_idx = []
        for i in range(self.num_of_videos):
            # avoid oversampling the sequence_frag
            for j in range(offset_count,
                           offset_count + (self.frames_in_trial[i] + 1 - self.sequence_frag_len),
                           self.frag_stride):
                valid_start_idx.append(j)
            offset_count += self.frames_in_trial[i]
        self.num_train_we_use = len(valid_start_idx)
        print('==> [RLLS Video Dataset batch] :: num train start idx RLLS12M: {:6d}'.format(self.num_train_we_use))
        print('==> [RLLS Video Dataset batch] :: num of all valid seq frag indices use: {:6d}'.format(
            self.num_train_we_use * self.sequence_frag_len))
        return valid_start_idx, self.num_train_we_use
