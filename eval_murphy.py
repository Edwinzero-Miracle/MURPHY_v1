import os
import argparse
import random
import time
import numpy as np

from utils.commons import str2bool


def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluate murphy online inference', add_help=False)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-g', '--gpu', default=[5], nargs='+', type=int, help='index of gpu to use, default 2')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--resume', type=str,
                        default='',
                        help='checkpoint name for current experiment')
    parser.add_argument('--model_type', type=str, default='ours14',
                        help='model type for training')
    parser.add_argument('--our_backbone', type=str, default='resnet50',
                        help='our model backbone for training: resnet50, resnet50lstm')
    parser.add_argument('--our_type', type=str, default='c2f_fuse_hrca',
                        help='our model type for training.')
    parser.add_argument('--our_enable_rgcn_module', default='true', type=str2bool,
                        help='enable relational GCN module, default False')
    parser.add_argument('--our_use_multilayer_gcn', default='true', type=str2bool,
                        help='enable 2-layered GCN, default True')
    parser.add_argument('--our_enable_adj_normalize', default='false', type=str2bool,
                        help='enable adjacency matrix normalization, default false')
    parser.add_argument('--our_disable_batch_type_sim', default='true', type=str2bool,
                        help='training only with the soft adjacency from backbone features, default False')
    parser.add_argument('--our_disable_gcn_lnorm', default='false', type=str2bool,
                        help='disable Layer Norm in MURPHY (must enable to achieve stability), default false')
    parser.add_argument('--our_use_training_connections', default='false', type=str2bool)
    parser.add_argument('--our_hrca_use_prior_know', default='true', type=str2bool,
                        help='use prior knowledge, default False')
    parser.add_argument('--ref_cnnlstm_hidden_dim', default=512, type=int,
                        help='ref cnnlstm hidden dimension, resnet50lstm with 512/256/128')
    parser.add_argument('--ref_cnnlstm_output_dim', default=512, type=int,
                        help='ref cnnlstm output dimension, resnet50lstm with 512/256/128')
    parser.add_argument('--our_use_pretrain_backbone_parameters', default='false', type=str2bool,
                        help='our_use_pretrain_backbone_parameters, default true')
    parser.add_argument('--our_hrca_use_rllc_connections', default='false', type=str2bool,
                        help='nesterov momentum, default False')
    parser.add_argument('--our_gcn_dim', default=256, type=int,
                        help='gcn feature output dimension, resnet50lstm with 512, swintrans with 128/256')
    parser.add_argument('--our_component_embedding_dim', default=512, type=int,
                        help='iao feature embedding dimension, resnet50lstm with 512/256, swintrans with 128/256')
    parser.add_argument('--our_hrca_hidden_dim', default=256, type=int,
                        help='HRCA hidden dimension, default is 128')
    parser.add_argument('--our_activation_func', type=str, default='relu',
                        help='our model activation for all layers: relu, lrelu(s = 0.2), elu')
    parser.add_argument('--our_optimizer', type=str, default='sgd',
                        help='our optimizer for training, sgd, adamW, mix(sgd+adamW), mix_trans ow [no_fuse] ...')

    parser.add_argument('--target_task', type=str,
                        default='step: 7, task: 16, triplet: 39, instrument: 12, action: 9, object: 17',
                        help='training task and corresponding number of classes, including the IDLE!,'
                             'step: 7, task: 16, step: 6+1, task: 15+1, triplet: 38+1, instrument: 11+1, action: 8+1, object: 16+1')
    parser.add_argument('--loss_weight', type=str,
                        default='loss_step:1.0, loss_task:1.0, loss_triplet:1.0, loss_instrument:1.0, loss_action:1.0, loss_object:1.0',
                        help='Weight for losses: loss_step:1.0, loss_triplet:1.0, loss_task:1.0'
                             ',loss_instrument:1.0, loss_action:1.0, loss_object:1.0')

    parser.add_argument('--loss_type', type=str, default='ce',
                        help='loss candidates: bce, ce')
    parser.add_argument('--crop_ui', default='false', type=str2bool,
                        help='crop the UI region in the image content, default to False')
    parser.add_argument('--shuffle_training_data', default='false', type=str2bool,
                        help='temporal GCN module needs true, o.w. default False')

    parser.add_argument('--debug_mode', default='deploy', type=str, help='dataset type for different purposes')
    parser.add_argument('--dataset', type=str, default='RLLS', help='The name of the dataset, RLLS')
    parser.add_argument('--difficulty_level', default='mix', type=str,
                        help='easy, medium, hard, mix. All splits a subject isolated')
    parser.add_argument('--split_id', default=1, type=int,
                        help='surgeon A (1), surgeon D(2), surgeon E(3)')

    parser.add_argument('--seq_length', default=2, type=int, help='sequence length, default 1')
    parser.add_argument('--frag_stride', default=1, type=int, help='non-overlap frames offset, default 1')
    parser.add_argument('--batch_size', default=16, type=int, help='train batch size, default 16')
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid batch size, default 16')
    parser.add_argument('-o', '--optimizer_choice', default=1, type=int, help='0 for sgd 1 for adam, default 1')
    parser.add_argument('-m', '--multi_optim', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=1, type=int, help='epochs to train and val, default 1')
    parser.add_argument('-w', '--num_workers', default=2, type=int, help='num of workers to use, default 2')
    parser.add_argument('-f', '--use_flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
    parser.add_argument('-c', '--crop_type', default=1, type=int,
                        help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
    parser.add_argument('-l', '--lr', default=1e-2, type=float, help='learning rate for optimizer')
    parser.add_argument('--lr_backbone', default=1e-2, type=float, help='learning rate for backbone')
    parser.add_argument('--lr_classifier', default=1e-2, type=float, help='learning rate for classifier')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for sgd, default 0')
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--damping', default=0, type=float, help='dampening for sgd, default 0')
    parser.add_argument('--use_nesterov', default=False, type=bool, help='nesterov momentum, default False')
    parser.add_argument('--sgd_adjust_lr', default=1, type=int,
                        help='sgd method adjust lr 0 for step 1 for min, default 1')
    parser.add_argument('--sgd_step', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
    parser.add_argument('--sgd_gamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
    parser.add_argument('-a', '--kl_alpha', default=1.0, type=float, help='kl loss ratio, default 1.0')

    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')

    parser.add_argument('--use_temporal_info', default=3, type=int,
                        help='temporal data loader')
    parser.add_argument('--dataset_directory', default='/home1/shangzhao/DPdatasets/MIRACLE_RLLS/', type=str,
                        help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')
    return parser


# SET ARGS AND GPU IDX
ap = argparse.ArgumentParser('RLLS test online script', parents=[get_args_parser()])
args_ = ap.parse_args()
gpu_usg = ",".join(list(map(str, args_.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg

import torch
from torch.nn import DataParallel

from models.murphy_net import MURPHYNet

from dataset import get_test_transform, get_training_split, build_test_dataset, build_test_dataloader
from modules.eval_one_epoch import evaluate
from modules.eval_one_epoch_frag import evaluate_frag_one_epoch
from modules.loss_module import build_criterion
from utils.checkpoint_saver import CheckpointSaver
from utils.summary_logger import TensorboardSummary

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def main(args):
    gpu_usg = ",".join(list(map(str, args_.gpu)))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
    use_gpu = torch.cuda.is_available()
    device = torch.device(args.device)
    args.device = device

    # Stage1 : Prepare dataset loaders
    test_transforms = get_test_transform()
    _, test_list, max_num_frames = get_training_split(args)
    test_dataset, iao_translation_dict = build_test_dataset(args, test_list, max_num_frames, test_transforms, 5)
    dataloader_test = build_test_dataloader(args, test_dataset, args.frag_stride)

    # Stage2 : Prepare model and pre-trained parameters
    classifier_types = {}
    for task in args.target_task.split(','):
        k, num_class = task.split(':')
        k = k.strip()
        num_class = int(num_class)
        classifier_types[k] = num_class
    if 'instrument' in classifier_types.keys() and \
            'action' in classifier_types.keys() and \
            'object' in classifier_types.keys():
        classifier_types['triplet'] = 39

    img_fdim = 512
    model = MURPHYNet(backbone=args.our_backbone, img_fdim=img_fdim,
                      gcn_dim=args.our_gcn_dim, out_dim=args.our_component_embedding_dim,
                      hrca_channels=args.our_hrca_hidden_dim, seq_len=5,
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
    print("( * ) ===> Training with CGNNnet [ours14]")

    model = DataParallel(model)
    if use_gpu:
        model = model.cuda()

    total_parameters = sum([p.numel() for n, p in model.named_parameters()])
    print('<*> Total number of params in [Network Model]:', f'{total_parameters / 1e6:,}M')

    if args.resume != '':
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        unexpected = [k for k in unexpected if 'running_mean' not in k and 'running_var' not in k]  # skip bn params
        # print("Unexpected keys: ", ','.join(unexpected))
        # check missing and unexpected keys
        if len(missing) > 0:
            # print("Missing keys: ", ','.join(missing))
            current_dict = model.state_dict()
            if 'module.rgcn.entry.weight' in missing and 'module.gcn.weight' in unexpected:
                current_dict['module.rgcn.entry.weight'] = pretrained_dict['module.gcn.weight']
                missing.remove('module.rgcn.entry.weight')
                unexpected.remove('module.gcn.weight')
            if 'module.rgcn.entry.bias' in missing and 'module.gcn.bias' in unexpected:
                current_dict['module.rgcn.entry.bias'] = pretrained_dict['module.gcn.bias']
                missing.remove('module.rgcn.entry.bias')
                unexpected.remove('module.gcn.bias')

        if len(missing) > 0:
            our_model = model.state_dict()
            for key, val in pretrained_dict.items():
                k = 'module.' + key
                if k in our_model.keys():
                    our_model[k] = val
                if k in missing:
                    missing.remove(k)
            model.load_state_dict(our_model)
            print("[Testing] :: Load pretrained parameter with adding module. prefix!")
            print("[Testing] ::      len(missing) = ", len(missing), " !!!!")

        if len(missing) != 0:
            print("Missing keys... check the keys of model!!!")
        print("Pre-trained model successfully loaded.")

    # initiate saver and logger
    checkpoint_saver = CheckpointSaver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)

    criterion = build_criterion(args, stage_training=False)
    print('Label type = ', args.target_task, ", label category number = ", classifier_types.values())

    # Stage3 : Prepare training logic
    print("Start testing")
    # validate
    if args.use_temporal_info == 1:
        evaluate(
            model, args.seq_length, classifier_types, dataloader_test, criterion, device,
            iao_translation_dict, 66666, summary_writer)
    else:
        evaluate_frag_one_epoch(
            model, args.seq_length, classifier_types, dataloader_test, criterion, device,
            iao_translation_dict, 66666, summary_writer)


if __name__ == '__main__':
    main(args_)
