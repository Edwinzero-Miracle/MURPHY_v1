import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import average_precision_score

np.seterr(invalid='ignore')
import warnings
warnings.filterwarnings("ignore")


def eval_AP_torch(pred: Tensor, gt: Tensor) -> dict:
    """

    :param pred: the category of prediction label
    :param gt: the category of ground truth label
    :return: the classwise AP and mAP
    """
    predicts = pred.detach().cpu().numpy()
    targets = gt.detach().cpu().numpy()

    class_ap = average_precision_score(targets, predicts, average=None)
    mean = np.nanmean(class_ap)
    return {"class_AP": class_ap, "mAP": mean}


def eval_AP_np(predicts: np.ndarray, targets: np.ndarray) -> dict:
    """

    :param predicts: the category of prediction label
    :param targets: the category of ground truth label (one-hot encoded)
    :return: the classwise AP and mAP
    """
    assert len(targets.shape) == 2
    class_ap = average_precision_score(targets.astype(np.float32), predicts, average=None)
    mean = np.nanmean(class_ap, axis=0)
    return {"class_AP": class_ap, "mAP": mean}


def eval_AP_nparray(predicts: np.ndarray, targets: np.ndarray, num_class: int) -> dict:
    """

    :param predicts: the category of prediction label
    :param targets: the category of ground truth label (true label)
    :return: the classwise AP and mAP
    """
    assert len(targets.shape) == 1

    def translate_label_to_onehot(arr: np.ndarray, num_class: int):
        return np.eye(num_class)[arr]

    tar_onehot = translate_label_to_onehot(targets, num_class).astype(np.float32)

    class_ap = average_precision_score(tar_onehot, predicts, average=None)
    mean = np.nanmean(class_ap, axis=0)
    return {"class_AP": class_ap, "mAP": mean}


def edit_distance(seq1, seq2):
    seq1 = [-1] + list(seq1)
    seq2 = [-1] + list(seq2)

    dist_matrix = np.zeros([len(seq1), len(seq2)], dtype=np.int)
    dist_matrix[:, 0] = np.arange(len(seq1))
    dist_matrix[0, :] = np.arange(len(seq2))

    for i in range(1, len(seq1)):
        for j in range(1, len(seq2)):
            if seq1[i] == seq2[j]:
                dist_matrix[i, j] = dist_matrix[i - 1, j - 1]
            else:
                operation_dists = [dist_matrix[i - 1, j],
                                   dist_matrix[i, j - 1],
                                   dist_matrix[i - 1, j - 1]]
                dist_matrix[i, j] = np.min(operation_dists) + 1

    return dist_matrix[-1, -1]


def segment_level(seq):
    segment_level_seq = []
    for label in seq.flatten():
        if len(segment_level_seq) == 0 or segment_level_seq[-1] != label:
            segment_level_seq.append(label)
    return segment_level_seq


def compute_edit_distance(prediction_seq, label_seq):
    """ Compute segment-level edit distance.
    First, transform each sequence to the segment level by replacing any
    repeated, adjacent labels with one label. Second, compute the edit distance
    (Levenshtein distance) between the two segment-level sequences.
    Simplified example: pretend each input sequence is only 1-D, with
    `prediction_seq = [1, 3, 2, 2, 3]` and `label_seq = [1, 2, 2, 2, 3]`.
    The segment-level equivalents are `[1, 3, 2, 3]` and `[1, 2, 3]`, resulting
    in an edit distance of 1.
    Args:
        prediction_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.
        label_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.
    Returns:
        A nonnegative integer, the number of operations () to transform the
        segment-level version of `prediction_seq` into the segment-level
        version of `label_seq`.
    """
    return edit_distance(segment_level(prediction_seq),
                         segment_level(label_seq))
