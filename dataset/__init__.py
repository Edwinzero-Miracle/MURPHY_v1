import torch.utils.data as data
from torchvision.transforms import transforms

from dataset.RLLS_split_config import get_training_mode_split
from dataset.RLLS_triplet_dataset import RLLSDataset
from dataset.RLLS_triplet_dataset_frags import RLLSVideoDataset, RLLSVideoSeqSampler

import numpy as np


def get_train_transform():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms


def get_test_transform():
    test_transforms = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return test_transforms


def get_training_split(args):
    if args.debug_mode == 'deploy':
        split_set = get_training_mode_split(config_name=args.difficulty_level, split_id=args.split_id)
        train_list = split_set['train']
        test_list = split_set['test']
        print(" Training list = ", train_list)
        print(" Testing list = ", test_list)
        max_num_frames = int(1e9)
    else:
        raise NotImplementedError

    return train_list, test_list, max_num_frames


def build_train_dataset(args, train_list, max_num_frames, train_transforms, fps: int = 5):
    if args.use_temporal_info == 1:
        train_dataset = RLLSDataset(args.dataset_directory, train_list, fps, max_num_frames, args.crop_ui,
                                    train_transforms)
        print("(warning) [RLLS_Triplet_Dataset] is applied for training/testing !")
    elif args.use_temporal_info == 3:
        train_dataset = RLLSVideoDataset(data_path=args.dataset_directory, trial_list=train_list,
                                         shuffle_in_batches=args.shuffle_training_data,
                                         sort_in_batches=args.sort_in_batches,
                                         sequence_frag_len=args.seq_length, default_fps=fps,
                                         max_num_frames=max_num_frames, crop_img_ui=args.crop_ui,
                                         transform=train_transforms)
        print("(warning) [RLLSVideoDataset] is applied for training ! "
              "** TEMP WINDOW SIZE ** = ", args.seq_length)
    else:
        raise NotImplementedError

    translation_dict = train_dataset.LABEL_IAO_CAT_TO_AA_CAT

    return train_dataset, translation_dict


def build_test_dataset(args, test_list, max_num_frames, test_transforms, fps: int = 5):
    if args.use_temporal_info == 1:
        test_dataset = RLLSDataset(args.dataset_directory, test_list, fps, max_num_frames, args.crop_ui,
                                   test_transforms)
        print("(warning) [RLLS_Triplet_Dataset] is applied for training/testing !")
    elif args.use_temporal_info == 3:
        test_dataset = RLLSVideoDataset(data_path=args.dataset_directory, trial_list=test_list,
                                        shuffle_in_batches=False,
                                        sort_in_batches=True,
                                        sequence_frag_len=args.seq_length, default_fps=fps,
                                        max_num_frames=max_num_frames, crop_img_ui=args.crop_ui,
                                        transform=test_transforms)
        print("(warning) [RLLSVideoDataset] is applied for testing ! "
              "** TEMP WINDOW SIZE ** = ", args.seq_length)
    else:
        raise NotImplementedError

    translation_dict = test_dataset.LABEL_IAO_CAT_TO_AA_CAT

    return test_dataset, translation_dict


def build_train_dataloader(args, train_dataset, frag_stride: int):
    print("[build_train_dataloader] :: (warnings) frag_stride is ", frag_stride, ", seq_len = ", args.seq_length)
    assert args.frag_stride == frag_stride
    if args.use_temporal_info == 1:
        data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=args.shuffle_training_data,
                                            num_workers=args.num_workers,
                                            pin_memory=True)
    elif args.use_temporal_info == 3:
        training_buffer_indices, num_train_we_use = train_dataset.prepare(args.seq_length, frag_stride)
        if args.shuffle_training_data:
            np.random.shuffle(training_buffer_indices)
        train_idx_shuffled = []
        for i in range(num_train_we_use):
            for j in range(args.seq_length):
                train_idx_shuffled.append(training_buffer_indices[i] + j)
        # Create the dataloader
        data_loader_train = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RLLSVideoSeqSampler(train_dataset, train_idx_shuffled),
            num_workers=args.num_workers,
            pin_memory=False)

    else:
        raise NotImplementedError
    return data_loader_train


def build_test_dataloader(args, test_dataset, frag_stride: int):
    assert args.frag_stride == frag_stride
    print("[build_train_dataloader] :: (warnings) frag_stride is ", frag_stride, ", seq_len = ", args.seq_length)
    print("[RLLSVideoDataset] :: (warnings) frag stride = ", frag_stride, " for testing")
    if args.use_temporal_info == 1:
        data_loader_test = data.DataLoader(test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
    elif args.use_temporal_info == 3:
        test_buffer_indices, num_test_we_use = test_dataset.prepare(args.seq_length, frag_stride)
        test_idx = []
        for i in range(num_test_we_use):
            for j in range(args.seq_length):
                test_idx.append(test_buffer_indices[i] + j)  # Create the dataloader
        data_loader_test = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=RLLSVideoSeqSampler(test_dataset, test_idx),
            num_workers=args.num_workers,
            pin_memory=False,
            prefetch_factor=2 * args.seq_length,
            persistent_workers=True
        )
    else:
        raise NotImplementedError
    return data_loader_test
