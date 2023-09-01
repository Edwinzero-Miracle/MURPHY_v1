
import json
import os
import pickle
from collections import defaultdict

import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


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


class RLLSDataset(Dataset):
    def __init__(self, data_path: str, trial_list: list, default_fps: int = 5, max_num_frames: int = 0,
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
        super(RLLSDataset, self).__init__()
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

        self.base_path = data_path
        self.json_dict_filename = self.base_path + TRIPLET_DICT_JSON
        self.load_label_dict(self.json_dict_filename)
        print("   [RLLSDataset :: Triplet]:: load dict file from: ", self.json_dict_filename, "  (True)")

        self.fps = default_fps
        self.trial_list = trial_list
        self.load_trial_data(self.trial_list, self.fps)

        self.loader = loader
        self.crop_img_ui = crop_img_ui

        self.transform = transform


        print('[RLLSDataset :: Triplet] num of properties  : {:6d} = 1 img, 6 labels'.format(len(self.data_buffer)))
        print('[RLLSDataset :: Triplet] num of frames  : {:6d}'.format(len(self.data_buffer['img_name'])))
        print('==> [RLLS Dataset] :: Load data successfully!, including: ', trial_list)

    def load_label_dict(self, json_dict_filename):
        if json_dict_filename is None:
            raise FileNotFoundError

        raw_label_dict = read_json(json_dict_filename)

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
                                         self.LABEL_ACT_TO_CAT[iao[1]], self.LABEL_OBJECT_TO_CAT[iao[2]]] = self.LABEL_AA_TO_CAT[aa]
            count += 1

        print("[RLLS_triplet_dataset.py] :: note :: Successfully load all surgical label dictionary ! ")

    def load_trial_data(self, trial_list: list, fps: int):
        self.trial_json_filenames = []
        for subject in trial_list:
            filename = self.base_path + SUBPATH_TO_LABEL + subject + '_fps' + str(fps) + '.json'
            if not os.path.exists(filename):
                print("Trial json file not exist: ", filename)
            self.trial_json_filenames.append(filename)
        print("total load json label trials: ", len(self.trial_json_filenames))

        # note: key =[image filename] ; value = [step, task, triplet, instrument, action, object]
        tmp_data = defaultdict(list)
        for i, filename in enumerate(self.trial_json_filenames):
            label_data = read_json(filename)

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

        self.data_buffer = defaultdict(list)
        if self.max_num_frames <= 0:
            self.max_num_frames = len(tmp_data.items())

        # You have to make sure the total samples can be divided by batch_size
        for img_name, tmp_labels in tmp_data.items():
            self.data_buffer['img_name'].append(self.base_path + SUBPATH_TO_IMAGE + str(self.fps) + 'fps/' + img_name)
            self.data_buffer['step'].append(tmp_labels[0])
            self.data_buffer['task'].append(tmp_labels[1])
            self.data_buffer['triplet'].append(tmp_labels[2])
            self.data_buffer['instrument'].append(tmp_labels[3])
            self.data_buffer['action'].append(tmp_labels[4])
            self.data_buffer['object'].append(tmp_labels[5])

        del tmp_data

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
        res['img_name'] = self.data_buffer['img_name'][index]

        return res

    def __len__(self):
        return len(self.data_buffer['img_name'])
