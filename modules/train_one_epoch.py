import math
import sys
import time
from typing import Iterable

import torch
from tqdm import tqdm
from utils.summary_logger import TensorboardSummary


def write_summary(stats, summary, epoch, mode):
    """
    write the current epoch result to tensorboard
    """
    summary.writer.add_scalar(mode + '/loss_step', stats['loss_step'], epoch)
    summary.writer.add_scalar(mode + '/loss_task', stats['loss_task'], epoch)
    summary.writer.add_scalar(mode + '/loss_triplet', stats['loss_triplet'], epoch)
    summary.writer.add_scalar(mode + '/loss_instrument', stats['loss_instrument'], epoch)
    summary.writer.add_scalar(mode + '/loss_action', stats['loss_action'], epoch)
    summary.writer.add_scalar(mode + '/loss_object', stats['loss_object'], epoch)


def train_one_epoch(
        model: torch.nn.Module, seq_length: int, classifier_types: dict,
        data_loader: Iterable, optimizer,
        criterion: torch.nn.Module, device: torch.device,
        epoch: int, summary: TensorboardSummary):
    """
    train model for 1 epoch
    """
    model.train()
    criterion.train()

    # initialize stats
    # shang: set the p2p, error_p2p, and total_p2p for loss value tracking
    train_stats = {'aggregated': 0.0, 'loss_idle': 0.0,
                   'loss_step': 0.0, 'loss_task': 0.0, 'loss_triplet': 0.0,
                   'loss_instrument': 0.0, 'loss_action': 0.0, 'loss_object': 0.0, 'debug_loss_instrument': 0.0}

    tbar = tqdm(data_loader, desc="[ Train -> Epoch: :%d ]" % epoch, ncols=80)
    start = time.time()
    num_imgs = len(data_loader.dataset)
    num_batches = len(data_loader)
    for idx, data in enumerate(tbar):
        # forward pass
        # inputs['img'] = b x c x h x w
        inputs = {'img': data['img'].to(torch.float).to(device),
                  'step': data['step'].to(device),
                  'task': data['task'].to(device),
                  'triplet': data['triplet'].to(device),
                  'instrument': data['instrument'].to(device),
                  'action': data['action'].to(device),
                  'object': data['object'].to(device)}
        inputs['idle'] = (inputs['triplet'] == 0).to(torch.long)

        # forward pass
        outputs = model(inputs)

        # compute loss
        losses = criterion(inputs, outputs)
        if losses is None:
            print("Loss value are invalid !!!! Bug Here !!!!")
            return train_stats

        # get the loss
        train_stats['aggregated'] += losses['aggregated'].item() * inputs['img'].size(0)

        if 'loss_step' in losses.keys():
            train_stats['loss_step'] += losses['loss_step'].item() * inputs['img'].size(0)
        if 'loss_task' in losses.keys():
            train_stats['loss_task'] += losses['loss_task'].item() * inputs['img'].size(0)
        if 'loss_triplet' in losses.keys():
            train_stats['loss_triplet'] += losses['loss_triplet'].item() * inputs['img'].size(0)
        if 'loss_instrument' in losses.keys():
            train_stats['loss_instrument'] += losses['loss_instrument'].item() * inputs['img'].size(0)
            train_stats['debug_loss_instrument'] += losses['loss_instrument'].item()
        if 'loss_action' in losses.keys():
            train_stats['loss_action'] += losses['loss_action'].item() * inputs['img'].size(0)
        if 'loss_object' in losses.keys():
            train_stats['loss_object'] += losses['loss_object'].item() * inputs['img'].size(0)

        # log for eval only
        # print('Index %d, aggregated %.4f, bce %.4f, ce %.4f, kl %.4f' %
        #       (idx, losses['aggregated'].item(), losses['bce'].item(), losses['ce'].item(), losses['kl'].item()))
        # print batch information
        # print("[ Batch index {:d} ] (time: {:.3f} s): total loss = {:.4f}, "
        #       "step loss = {:.4f}, task loss = {:.4f}, triplet loss = {:.4f},\n"
        #       "instrument loss = {:.4f}, action loss = {:.4f}, object loss = {:.4f}".format(
        #     idx, (time.time() - start), losses['aggregated'].item(), losses['loss_step'].item(),
        #     losses['loss_task'].item(), losses['loss_triplet'].item(),
        #     losses['loss_instrument'].item(), losses['loss_action'].item(), losses['loss_object'].item()))

        # terminate training if exploded
        if not math.isfinite(losses['aggregated'].item()):
            print('Loss is {}, stopping training'.format(losses['aggregated'].item()))
            sys.exit(1)

        # backprop
        if isinstance(optimizer, dict):
            for key, opt in optimizer.items():
                opt.zero_grad()
        else:
            optimizer.zero_grad()

        losses['aggregated'].backward()

        # clip norm
        max_norm = 5.0
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # step optimizer
        if isinstance(optimizer, dict):
            for key, opt in optimizer.items():
                opt.step()
        else:
            optimizer.step()


        # # clear cache
        # torch.cuda.empty_cache()

        # print batch information
        # print("[ Batch index {:d} ] (time: {:.3f} s): total loss = {:.4f}, "
        #       "step loss = {:.4f}, task loss = {:.4f}, triplet loss = {:.4f},\n"
        #       "instrument loss = {:.4f}, action loss = {:.4f}, object loss = {:.4f}".format(
        #     idx, (time.time() - start), losses['aggregated'].item(), losses['loss_step'].item(),
        #     losses['loss_task'].item(), losses['loss_triplet'].item(),
        #     losses['loss_instrument'].item(), losses['loss_action'].item(), losses['loss_object'].item()))

    print()
    print(
        "==> [ Epoch {:d} ] (time: {:.3f} s): average total loss = {:.6f}, "
        "step loss = {:.6f}, task loss = {:.6f}, triplet loss = {:.6f}, "
        "instrument loss = {:.6f}, action loss = {:.6f}, object loss = {:.6f}".format(
            epoch, (time.time() - start),
            train_stats['aggregated'] / num_imgs,
            train_stats['loss_step'] / num_imgs,
            train_stats['loss_task'] / num_imgs,
            train_stats['loss_triplet'] / num_imgs,
            train_stats['loss_instrument'] / num_imgs,
            train_stats['loss_action'] / num_imgs,
            train_stats['loss_object'] / num_imgs))

    # log to tensorboard
    write_summary(train_stats, summary, epoch, 'train')
    return train_stats
