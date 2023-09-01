import time
from typing import Iterable

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.summary_logger import TensorboardSummary

from modules.evaluation_metric import eval_AP_np


def write_summary(stats, summary, epoch, mode):
    summary.writer.add_scalar(mode + '/loss_step', stats['loss_step'], epoch)
    summary.writer.add_scalar(mode + '/loss_task', stats['loss_task'], epoch)
    summary.writer.add_scalar(mode + '/loss_triplet', stats['loss_triplet'], epoch)
    summary.writer.add_scalar(mode + '/loss_instrument', stats['loss_instrument'], epoch)
    summary.writer.add_scalar(mode + '/loss_action', stats['loss_action'], epoch)
    summary.writer.add_scalar(mode + '/loss_object', stats['loss_object'], epoch)

    summary.writer.add_scalar(mode + '/map_step', stats['map_step'], epoch)
    summary.writer.add_scalar(mode + '/map_task', stats['map_task'], epoch)
    summary.writer.add_scalar(mode + '/map_triplet', stats['map_triplet'], epoch)
    summary.writer.add_scalar(mode + '/map_instrument', stats['map_instrument'], epoch)
    summary.writer.add_scalar(mode + '/map_action', stats['map_action'], epoch)
    summary.writer.add_scalar(mode + '/map_object', stats['map_object'], epoch)


@torch.no_grad()
def evaluate_frag_one_epoch(
        model: torch.nn.Module, seq_length: int, classifier_types: dict,
        data_loader: Iterable,
        criterion: torch.nn.Module, device: torch.device,
        translation_dictionary: dict,
        epoch: int, summary: TensorboardSummary,
        save_each_sample=False):
    """
    eval model for 1 epoch

    :param classifier_types:
    :param seq_length:
    :param model:
    :param data_loader:
    :param device:
    :param epoch:
    :param summary:
    :param criterion:
    :param save_each_sample: The flag to save all entries for fast evaluation
    """
    model.eval()
    criterion.eval()

    # initialize stats
    # shang: set the p2p, error_p2p, and total_p2p for loss value tracking
    eval_stats = {'aggregated': 0.0, 'loss_idle': 0.0, 'loss_step': 0.0, 'loss_task': 0.0,
                  'loss_triplet': 0.0, 'loss_instrument': 0.0, 'loss_action': 0.0, 'loss_object': 0.0,
                  'map_step': 0.0, 'map_task': 0.0, 'map_triplet': 0.0,
                  'map_instrument': 0.0, 'map_action': 0.0, 'map_object': 0.0,
                  }

    predicts = {
        "idle": [],
        "step": [],
        "task": [],
        "triplet": [],
        "instrument": [],
        "action": [],
        "object": []
    }
    targets = {
        "idle": [],
        "step": [],
        "task": [],
        "triplet": [],
        "instrument": [],
        "action": [],
        "object": []
    }

    tbar = tqdm(data_loader, desc="[ Eval -> Epoch: :%d ]" % epoch, ncols=80)
    start = time.time()
    num_imgs = data_loader.dataset.total_frames
    for idx, data in enumerate(tbar):
        # forward pass
        B, C, H, W = data['img'].shape
        inputs = {'img': data['img'].view(-1, seq_length, C, H, W).to(torch.float).to(device),  # we need to squeeze the
                  'step': data['step'][(seq_length - 1)::seq_length].to(device),
                  'task': data['task'][(seq_length - 1)::seq_length].to(device),
                  'triplet': data['triplet'][(seq_length - 1)::seq_length].to(device),
                  'instrument': data['instrument'][(seq_length - 1)::seq_length].to(device),
                  'action': data['action'][(seq_length - 1)::seq_length].to(device),
                  'object': data['object'][(seq_length - 1)::seq_length].to(device)}

        inputs['idle'] = (inputs['triplet'] == 0).to(torch.long)

        # forward pass
        outputs = model(inputs)

        # compute loss
        losses = criterion(inputs, outputs)
        if losses is None:
            print("Loss value are invalid !!!! Bug Here !!!!")
            return eval_stats
        eval_stats['aggregated'] += losses['aggregated'].item() * inputs['img'].size(0)

        gt_tensors = {}
        for key, value in classifier_types.items():
            gt_tensors[key] = F.one_hot(inputs[key], value)

        if 'loss_step' in losses.keys():
            eval_stats['loss_step'] += losses['loss_step'].item() * inputs['img'].size(0)
            predicts['step'].append(outputs['step'].detach().cpu().numpy())
            targets['step'].append(gt_tensors['step'].detach().cpu().numpy())
        if 'loss_task' in losses.keys():
            eval_stats['loss_task'] += losses['loss_task'].item() * inputs['img'].size(0)
            predicts['task'].append((outputs['task']).detach().cpu().numpy())
            targets['task'].append(gt_tensors['task'].detach().cpu().numpy())
        if 'loss_triplet' in losses.keys():
            eval_stats['loss_triplet'] += losses['loss_triplet'].item() * inputs['img'].size(0)
            predicts['triplet'].append((outputs['triplet']).detach().cpu().numpy())
            targets['triplet'].append(gt_tensors['triplet'].detach().cpu().numpy())
        if 'loss_instrument' in losses.keys():
            eval_stats['loss_instrument'] += losses['loss_instrument'].item() * inputs['img'].size(0)
            predicts['instrument'].append((outputs['instrument']).detach().cpu().numpy())
            targets['instrument'].append(gt_tensors['instrument'].detach().cpu().numpy())
        if 'loss_action' in losses.keys():
            eval_stats['loss_action'] += losses['loss_action'].item() * inputs['img'].size(0)
            predicts['action'].append((outputs['action']).detach().cpu().numpy())
            targets['action'].append(gt_tensors['action'].detach().cpu().numpy())
        if 'loss_object' in losses.keys():
            eval_stats['loss_object'] += losses['loss_object'].item() * inputs['img'].size(0)
            predicts['object'].append((outputs['object']).detach().cpu().numpy())
            targets['object'].append(gt_tensors['object'].detach().cpu().numpy())

        # clear cache
        torch.cuda.empty_cache()

    if len(predicts['step']):
        step_pred = np.concatenate(([i for i in predicts['step']]), axis=0)
        step_tar = np.concatenate(([t for t in targets['step']]), axis=0)

        eval_step = eval_AP_np(step_pred, step_tar)
        eval_stats['map_step'] = eval_step['mAP']

    if len(predicts['task']):
        task_pred = np.concatenate(([i for i in predicts['task']]), axis=0)
        task_tar = np.concatenate(([t for t in targets['task']]), axis=0)

        eval_task = eval_AP_np(task_pred, task_tar)
        eval_stats['map_task'] = eval_task['mAP']

    if len(predicts['triplet']):
        triplet_pred = np.concatenate(([i for i in predicts['triplet']]), axis=0)
        triplet_tar = np.concatenate(([t for t in targets['triplet']]), axis=0)

        eval_triplet = eval_AP_np(triplet_pred, triplet_tar)
        eval_stats['map_triplet'] = eval_triplet['mAP']

    if len(predicts['instrument']):
        instrument_pred = np.concatenate(([i for i in predicts['instrument']]), axis=0)
        instrument_tar = np.concatenate(([t for t in targets['instrument']]), axis=0)

        eval_instrument = eval_AP_np(instrument_pred, instrument_tar)
        eval_stats['map_instrument'] = eval_instrument['mAP']

    if len(predicts['action']):
        action_pred = np.concatenate(([i for i in predicts['action']]), axis=0)
        action_tar = np.concatenate(([t for t in targets['action']]), axis=0)

        eval_action = eval_AP_np(action_pred, action_tar)
        eval_stats['map_action'] = eval_action['mAP']

    if len(predicts['object']):
        object_pred = np.concatenate(([i for i in predicts['object']]), axis=0)
        object_tar = np.concatenate(([t for t in targets['object']]), axis=0)

        eval_object = eval_AP_np(object_pred, object_tar)
        eval_stats['map_object'] = eval_object['mAP']

    write_summary(eval_stats, summary, epoch, 'eval')
    # log to text
    print(
        "==> [ EvalLoss @ Epoch {:d} ][True] (time: {:.3f} s): average total loss = {:.6f}, "
        "step loss = {:.6f}, task loss = {:.6f}, triplet loss = {:.6f}, "
        "instrument loss = {:.6f}, action loss = {:.6f}, object loss = {:.6f}".format(
            epoch, (time.time() - start),
            eval_stats['aggregated'] / num_imgs,
            eval_stats['loss_task'] / num_imgs,
            eval_stats['loss_step'] / num_imgs,
            eval_stats['loss_triplet'] / num_imgs,
            eval_stats['loss_instrument'] / num_imgs,
            eval_stats['loss_action'] / num_imgs,
            eval_stats['loss_object'] / num_imgs))
    print('==> [Epoch %d : Time (%.3f) s], [MAP EVALUATION of %s] :: \n'
          '[*][True] map_step %.4f, map_task = %.4f, map_triplet = %.4f,'
          '{ map_instrument = %.4f, map_action = %.4f, map_object = %.4f }' %
          (epoch, (time.time() - start), classifier_types.keys(),
           eval_stats['map_step'], eval_stats['map_task'],
           eval_stats['map_triplet'],
           eval_stats['map_instrument'], eval_stats['map_action'], eval_stats['map_object']))

    return eval_stats
