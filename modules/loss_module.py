import math
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class SVACriterion(nn.Module):
    """
        SVA: Surgical Video Analysis
    """

    def __init__(self, classifier_dict: dict, stage_training: bool,
                 loss_type: str, loss_weight: dict = None):
        super(SVACriterion, self).__init__()

        if loss_weight is None:
            loss_weight = {}

        self.epsilon = 1 - math.log(2)

        self.stage_training = stage_training
        self.weights = loss_weight
        self.classifier_dict = classifier_dict

        self.loss_type = loss_type
        if self.loss_type == "bce":
            self.step_criterion = nn.BCEWithLogitsLoss()
            self.task_criterion = nn.BCEWithLogitsLoss()
            self.triplet_criterion = nn.BCEWithLogitsLoss()

            self.instrument_criterion = nn.BCEWithLogitsLoss()
            self.action_criterion = nn.BCEWithLogitsLoss()
            self.object_criterion = nn.BCEWithLogitsLoss()

        elif self.loss_type == "ce":
            self.step_criterion = nn.CrossEntropyLoss()
            self.task_criterion = nn.CrossEntropyLoss()
            self.triplet_criterion = nn.CrossEntropyLoss()

            self.instrument_criterion = nn.CrossEntropyLoss()
            self.action_criterion = nn.CrossEntropyLoss()
            self.object_criterion = nn.CrossEntropyLoss()
        else:
            print("[SVACriterion] :: unsupported loss function, please implement it!")
            raise NotImplementedError

    def forward(self, ground_truth: dict, pred: dict):
        """
        :param pred: input data, dict of tensors
        :param ground_truth: label data, dict of the gt category (int)
        :return: loss dictionary
        """
        loss = {}

        if pred.keys() is None:
            return

        if pred['step'] is not None:
            if self.loss_type == "bce":
                loss['loss_step'] = self.step_criterion(pred['step'],
                                                       F.one_hot(ground_truth['step'].to(torch.long),
                                                                 self.classifier_dict['step']).float())
            elif self.loss_type == 'ce':
                loss['loss_step'] = self.step_criterion(pred['step'], ground_truth['step'])
            else:
                raise NotImplementedError

        if pred['task'] is not None:
            if self.loss_type == "bce":
                loss['loss_task'] = self.task_criterion(pred['task'],
                                                       F.one_hot(ground_truth['task'],
                                                                 self.classifier_dict['task']).float())
            elif self.loss_type == 'ce':
                loss['loss_task'] = self.task_criterion(pred['task'], ground_truth['task'])
            else:
                raise NotImplementedError

        if pred['triplet'] is not None:
            if self.loss_type == "bce":
                loss['loss_triplet'] = self.triplet_criterion(pred['triplet'],
                                                             F.one_hot(ground_truth['triplet'],
                                                                       self.classifier_dict['triplet']).float())
            elif self.loss_type == 'ce':
                loss['loss_triplet'] = self.triplet_criterion(pred['triplet'], ground_truth['triplet'])
            else:
                raise NotImplementedError

        if pred['instrument'] is not None:
            if self.loss_type == "bce":
                loss['loss_instrument'] = self.instrument_criterion(pred['instrument'],
                                                                   F.one_hot(ground_truth['instrument'],
                                                                             self.classifier_dict[
                                                                                 'instrument']).float())
            elif self.loss_type == 'ce':
                loss['loss_instrument'] = self.instrument_criterion(pred['instrument'], ground_truth['instrument'])
            else:
                raise NotImplementedError

        if pred['action'] is not None:
            if self.loss_type == "bce":
                loss['loss_action'] = self.action_criterion(pred['action'],
                                                           F.one_hot(ground_truth['action'],
                                                                     self.classifier_dict['action']).float())
            elif self.loss_type == 'ce':
                loss['loss_action'] = self.action_criterion(pred['action'], ground_truth['action'])
            else:
                raise NotImplementedError

        if pred['object'] is not None:
            if self.loss_type == "bce":
                loss['loss_object'] = self.object_criterion(pred['object'],
                                                           F.one_hot(ground_truth['object'],
                                                                     self.classifier_dict['object']).float())
            elif self.loss_type == 'ce':
                loss['loss_object'] = self.object_criterion(pred['object'], ground_truth['object'])
            else:
                raise NotImplementedError

        total_loss = 0.0
        for key in self.weights.keys():
            if self.stage_training is True and 'idle' in key:
                continue
            total_loss += loss[key] * self.weights[key]

        loss['aggregated'] = total_loss
        return OrderedDict(loss)


def build_criterion(args, stage_training: bool):
    loss_weight = {}
    for weight in args.loss_weight.split(','):
        k, v = weight.split(':')
        k = k.strip()
        v = float(v)
        loss_weight[k] = v

    classifier_types = {}
    for task in args.target_task.split(','):
        k, num_class = task.split(':')
        k = k.strip()
        num_class = int(num_class)
        classifier_types[k] = num_class

    print("[Build Criterion] :: loss type is: ", args.loss_type)

    return SVACriterion(classifier_types, stage_training, args.loss_type, loss_weight)
