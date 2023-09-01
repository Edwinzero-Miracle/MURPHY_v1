import os
import argparse
import random
import time
import numpy as np

from utils.commons import str2bool

np.seterr(divide='ignore', invalid='ignore')
import warnings

warnings.filterwarnings("ignore")

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)


def get_args_parser():
    parser = argparse.ArgumentParser(description='baseline gesture recognition training', add_help=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-g', '--gpu', default=[6], nargs='+', type=int, help='index of gpu to use, default 2')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seq_length', default=1, type=int, help='sequence length, default 1')
    parser.add_argument('--frag_stride', default=1, type=int, help='non-overlap frames offset, default 1')
    parser.add_argument('--batch_size', default=16, type=int, help='train batch size, default 16')
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid batch size, default 16')
    parser.add_argument('-o', '--optimizer_choice', default=1, type=int, help='0 for sgd 1 for adam, default 1')
    parser.add_argument('-m', '--multi_optim', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=51, type=int, help='epochs to train and val, default 51')
    parser.add_argument('-w', '--num_workers', default=4, type=int, help='num of workers to use, default 2')
    parser.add_argument('-f', '--use_flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
    parser.add_argument('-c', '--crop_type', default=1, type=int,
                        help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')

    parser.add_argument('-l', '--lr', default=1e-2, type=float, help='learning rate for optimizer, default 1e-2')
    parser.add_argument('--lr_backbone', default=1e-2, type=float, help='learning rate for backbone')
    parser.add_argument('--lr_gcn', default=1e-2, type=float, help='learning rate for rgcn')
    parser.add_argument('--lr_classifier', default=1e-2, type=float, help='learning rate for classifier')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for sgd, default 0')
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--loss_type', type=str, default='bce', help='loss candidates: bce, ce')
    parser.add_argument('--damping', default=0, type=float, help='dampening for sgd, default 0')
    parser.add_argument('--use_nesterov', default='False', type=str2bool, help='nesterov momentum, default False')
    parser.add_argument('--sgd_adjust_lr', default=1, type=int,
                        help='sgd method adjust lr 0 for step 1 for min, default 1')
    parser.add_argument('--sgd_step', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
    parser.add_argument('--sgd_gamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
    parser.add_argument('-a', '--kl_alpha', default=1.0, type=float, help='kl loss ratio, default 1.0')

    parser.add_argument('--model_type', type=str, default='ours14',
                        help='model type for training: murphy')
    parser.add_argument('--crop_ui', default='false', type=str2bool,
                        help='crop the UI region in the image content, default to False')
    parser.add_argument('--shuffle_training_data', default='true', type=str2bool,
                        help='temporal GCN module needs true, o.w. default False')
    parser.add_argument('--sort_in_batches', default='false', type=str2bool)

    parser.add_argument('--our_enable_rgcn_module', default='true', type=str2bool,
                        help='enable relational GCN module, default False')
    parser.add_argument('--our_use_multilayer_gcn', default='true', type=str2bool)
    parser.add_argument('--our_enable_adj_normalize', default='false', type=str2bool,
                        help='enable adjacency matrix normalization')
    parser.add_argument('--our_disable_batch_type_sim', default='false', type=str2bool,
                        help='training only with the soft adjacency from backbone features, default False')
    parser.add_argument('--our_disable_gcn_lnorm', default='false', type=str2bool,
                        help='disable Layer Norm in MURPHY (must enable to achieve stability), default false')
    parser.add_argument('--our_use_training_connections', default='true', type=str2bool)
    parser.add_argument('--our_gcn_dim', default=256, type=int,
                        help='gcn feature output dimension')
    parser.add_argument('--our_component_embedding_dim', default=512, type=int,
                        help='iao feature embedding dimension')
    parser.add_argument('--our_hrca_hidden_dim', default=512, type=int,
                        help='HRCA hidden dimension, default is 128')
    parser.add_argument('--our_activation_func', type=str, default='relu',
                        help='our model activation for all layers: relu, lrelu(s = 0.2), elu')

    parser.add_argument('--our_type', type=str, default='c2f_fuse_hrca',
                        help='our model type for training')
    parser.add_argument('--our_optimizer', type=str, default='sgd',
                        help='our optimizer for training, sgd, adam ...')

    parser.add_argument('--our_hrca_use_prior_know', default='true', type=str2bool,
                        help='use prior knowledge, default False')
    parser.add_argument('--our_hrca_use_rllc_connections', default='true', type=str2bool,
                        help='use rllc connections, default False')
    parser.add_argument('--our_use_pretrain_backbone_parameters', default='false', type=str2bool,
                        help='use rllc connections, default False')

    parser.add_argument('--our_backbone', type=str, default='resnet50',
                        help='our model backbone for training: resnet50, resnet50lstm')
    parser.add_argument('--our_freeze_backbone', default='false', type=str2bool,
                        help='freeze backbone during training, default False')
    parser.add_argument('--pretrained_backbone',
                        default='',
                        help='pre-trained parameters from pre-training checkpoint')
    parser.add_argument('--resume',
                        default='',
                        help='resume from checkpoint')

    parser.add_argument('--target_task', type=str,
                        default='step: 7, task: 16, triplet: 39, instrument: 12, action: 9, object: 17',
                        help='training task and corresponding number of classes, including the IDLE!,'
                             'step: 6+1, task: 15+1, triplet: 38+1, instrument: 11+1, action: 8+1, object: 16+1')
    parser.add_argument('--loss_weight', type=str,
                        default='loss_step:1.0, loss_task: 1.0, loss_triplet:1.0, loss_instrument:1.0, loss_action:1.0, loss_object:1.0',
                        help='Weight for losses')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')

    parser.add_argument('--use_temporal_info', default=1, type=int,
                        help='temporal data loader')
    parser.add_argument('--debug_mode', default='deploy', type=str, help='dataset type for different purposes')
    parser.add_argument('--dataset', type=str, default='RLLS', help='The name of the dataset, RLLS')
    parser.add_argument('--difficulty_level', default='mix', type=str,
                        help='easy, medium, hard, mix. All splits a subject isolated')
    parser.add_argument('--split_id', default=1, type=int,
                        help='surgeon A (1), surgeon D(2), surgeon E(3)')
    parser.add_argument('--dataset_directory', default='/home1/shangzhao/DPdatasets/MIRACLE_RLLS/', type=str,
                        help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')
    return parser


ap = argparse.ArgumentParser('RLLS training script', parents=[get_args_parser()])
args_ = ap.parse_args()

gpu_usg = ",".join(list(map(str, args_.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg

import torch
import torch.nn as nn

from models.murphy_net import MURPHYNet

from dataset import get_train_transform, get_training_split, \
    build_train_dataset, build_train_dataloader
from modules.loss_module import build_criterion
from modules.train_one_epoch import train_one_epoch
from utils.checkpoint_saver import CheckpointSaver
from utils.summary_logger import TensorboardSummary

torch.set_num_threads(cpu_num)


def print_model_param(model: nn.Module):
    """
    print number of parameters in the model
    """
    total_parameters = sum([p.numel() for n, p in model.named_parameters()])
    print('<*> Total number of params in [Network Model]:', f'{total_parameters / 1e6:,}M')


def save_checkpoint(epoch, model, checkpoint_saver):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'state_dict': model.state_dict()
    }
    checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)


def main(args):
    num_gpu = torch.cuda.device_count()
    use_gpu = torch.cuda.is_available()
    device = torch.device(args.device)
    args.device = device

    print('number of gpu   : {:6d}'.format(num_gpu))
    print('sequence length : {:6d}'.format(args.seq_length))
    print('train batch size: {:6d}'.format(args.batch_size))
    print('test batch size: {:6d}'.format(args.test_batch_size))
    print('optimizer choice: {:6d}'.format(args.optimizer_choice))
    print('multiple optim  : {:6d}'.format(args.multi_optim))
    print('num of epochs   : {:6d}'.format(args.epochs))
    print('num of workers  : {:6d}'.format(args.num_workers))
    print('whether to flip : {:6d}'.format(args.use_flip))
    print('learning rate   : {:.4f}'.format(args.lr))

    # Stage1 : Prepare dataset loaders
    train_transforms = get_train_transform()
    train_list, _, max_num_frames = get_training_split(args)
    train_dataset, iao_translation_dict = build_train_dataset(args, train_list, max_num_frames, train_transforms, 5)
    dataloader_train = build_train_dataloader(args, train_dataset, frag_stride=args.frag_stride)

    # Stage2 : Prepare model and pre-trained parameters
    classifier_types = {}
    for task in args.target_task.split(','):
        k, num_class = task.split(':')
        k = k.strip()
        num_class = int(num_class)
        classifier_types[k] = num_class

    img_fdim = 512
    model = MURPHYNet(backbone=args.our_backbone, img_fdim=img_fdim,
                      gcn_dim=args.our_gcn_dim, out_dim=args.our_component_embedding_dim,
                      hrca_channels=args.our_hrca_hidden_dim,
                      num_instrument=12, num_action=9, num_object=17,
                      num_triplets=39, num_task=16, num_step=7,
                      type=args.our_type,
                      relations=classifier_types,
                      use_rlls_rc_mode=args.our_hrca_use_rllc_connections,
                      use_multi_layer_gcn=args.our_use_multilayer_gcn,
                      enable_adj_normalize=args.our_enable_adj_normalize,
                      disable_batch_type_sim=args.our_disable_batch_type_sim,
                      use_prior_knowledge=args.our_hrca_use_prior_know,
                      activation=args.our_activation_func)
    if len(args.pretrained_backbone) > 0:
        model.load_pretrained_backbone(args.pretrained_backbone)
    print("( * ) ===> Training with MURPHYNet [ours14]")

    print_model_param(model)
    # model = DataParallel(model)
    if use_gpu:
        model = model.cuda()

    # load checkpoint if provided
    if args.resume != '':
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        current_model_dict = model.state_dict()
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), pretrained_dict.values())}
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        count = 0
        if len(missing) > 0:
            our_model_dict = model.state_dict()
            for k, v in pretrained_dict.items():
                if k in our_model_dict.keys() and v.shape == our_model_dict[k].shape:
                    # check if pretrain raw works
                    our_model_dict[k] = v
                    count = count + 1
                else:
                    name = k[7:]  # remove `module.` and check again
                    if name in our_model_dict.keys():
                        # print(name)
                        our_model_dict[name] = v
                        count = count + 1
            model.load_state_dict(our_model_dict)
            print("[Training] :: total pretrained_backbone copied: ", count)

        # check missing and unexpected keys
        # if len(missing) > count:
        #       print("Still Missing keys: ", ','.join(missing))
        #       raise Exception("Missing keys.")

        # unexpected = [k for k in unexpected if 'running_mean' not in k and 'running_var' not in k]  # skip bn params
        # if len(unexpected) > 0 and 'ours' not in args.model_type:
        # print("Unexpected keys: ", ','.join(unexpected))
        print("Pre-trained model successfully loaded.")
        print("Loading from: ", args.resume)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if
                    "backbone" not in n and "gcn" not in n and "hrca" not in n and "classifier" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "gcn" in n and "hrca" in n and p.requires_grad],
            "lr": args.lr_gcn,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad],
            "lr": args.lr_classifier,
        },
    ]

    if args.our_optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
        print("[Training] :: All parts use SGD ! (True)")
        print("[Training] ::     lr = {:6f}, lr_backbone = {:6f}, lr_gcn = {:6f}, lr_classifier = {:6f}! ".format(
            args.lr, args.lr_backbone, args.lr_gcn, args.lr_classifier
        ))
    elif args.our_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
        print("[Training] :: All parts use ADAM ! (True)")
        print("[Training] ::     lr = {:6f}, lr_backbone = {:6f}, lr_gcn = {:6f}, lr_classifier = {:6f}! ".format(
            args.lr, args.lr_backbone, args.lr_gcn, args.lr_classifier
        ))
    else:
        raise NotImplementedError

    # initiate saver and logger
    checkpoint_saver = CheckpointSaver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)

    total_param_str = "Total number of params in [Network Model] = " + \
                      f'{sum([p.numel() for n, p in model.named_parameters()]) / 1e6:,}M'
    checkpoint_saver.save_experiment_epoch_log(total_param_str)

    criterion = build_criterion(args, stage_training=False)
    print('Label type = ', args.target_task, ", label category number = ", classifier_types.values())

    # Stage3 : Prepare training logic
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch: {}/{}'.format(epoch, args.epochs - 1))

        if args.use_temporal_info == 1:
            train_one_epoch(model, args.batch_size, classifier_types, dataloader_train, optimizer,
                            criterion, args.device, epoch, summary_writer)
        else:
            raise NotImplementedError

        if not args.pre_train:
            if isinstance(lr_scheduler, dict):
                for key, lr_sch in lr_scheduler.items():
                    lr_sch.step()
                    print(key, " :: => current learning rate", lr_sch.get_lr())
            else:
                lr_scheduler.step()
                print("current learning rate", lr_scheduler.get_lr())

        # empty cache
        torch.cuda.empty_cache()

        # save if pretrain, save every 2 epochs
        if args.pre_train or epoch % 2 == 0:
            save_checkpoint(epoch, model, checkpoint_saver)

    # save final model
    save_checkpoint(epoch, model, checkpoint_saver)
    print("[* _ *] Experiment path: ", checkpoint_saver.experiment_dir)


if __name__ == '__main__':
    main(args_)
