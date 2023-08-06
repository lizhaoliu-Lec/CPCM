"""
Reference: https://github.com/liuzhengzhe/One-Thing-One-Click/blob/master/3D-U-Net/util/eval.py
"""
import argparse
import logging
import time
import os
from prettytable import PrettyTable
import numpy as np
import torch.distributed as dist
import torch
from fvcore.common.registry import Registry

import util.utils as util
from util.utils import AverageMeter
import util.utils_3d as util_3d

try:
    from itertools import izip
except ImportError:
    izip = zip

# overlaps for evaluation
from meta_data.constant import PROJECT_NAME
from meta_data.scan_net import CLASS_LABELS, VALID_CLASS_IDS, ID_TO_LABEL, CLASS2Id

OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
MIN_REGION_SIZES = np.array([100])
# distance thresholds [m]
DISTANCE_THRESHES = np.array([float('inf')])
# distance confidences
DISTANCE_CONFS = np.array([-float('inf')])


def evaluate_matches(matches):
    overlaps = OVERLAPS
    min_region_sizes = [MIN_REGION_SIZES[0]]
    dist_threshes = [DISTANCE_THRESHES[0]]
    dist_confs = [DISTANCE_CONFS[0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), np.float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt[
                                        'med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                    gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise, set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                                    gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt[
                                    'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    if len(y_true_sorted_cumsum) == 0:
                        num_true_examples = 0
                    else:
                        num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(OVERLAPS, 0.5))
    o25 = np.where(np.isclose(OVERLAPS, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(OVERLAPS, 0.25)))
    avg_dict = {'all_ap': np.nanmean(aps[d_inf, :, oAllBut25]),
                'all_ap_50%': np.nanmean(aps[d_inf, :, o50]),
                'all_ap_25%': np.nanmean(aps[d_inf, :, o25]),
                "classes": {}}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
    return avg_dict


def assign_instances_for_scan(scene_name, pred_info, gt_file):
    gt_ids = None
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))
    assert gt_ids is not None

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    # associate
    gt2pred = gt_instances.copy()
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    nMask = pred_info['label_id'].shape[0]
    for i in range(nMask):
        label_id = int(pred_info['label_id'][i])
        conf = pred_info['conf'][i]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info['mask'][i]  # (N), long
        if len(pred_mask) != len(gt_ids):
            util.print_error('wrong number of lines in mask#%d: ' % (i) + '(%d) vs #mesh vertices (%d)' % (
                len(pred_mask), len(gt_ids)))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < MIN_REGION_SIZES[0]:
            continue  # skip if empty

        pred_instance = {
            'filename': '{}_{:03d}'.format(scene_name, num_pred_instances),
            'pred_id': num_pred_instances,
            'label_id': label_id,
            'vert_count': num,
            'confidence': conf,
            'void_intersection': np.count_nonzero(np.logical_and(bool_void, pred_mask))
        }

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs):
    logger = logging.getLogger(PROJECT_NAME)
    sep = ""
    col1 = ":"
    lineLen = 64

    logger.info("")
    logger.info("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    logger.info(line)
    logger.info("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        logger.info(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    logger.info("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    logger.info(line)
    logger.info("")


def calculate_remain_time(cur_step, total_step, avg_time):
    remain_time = (total_step - cur_step) * avg_time
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return remain_time


EVALUATOR_REGISTRY = Registry("Evaluator")
EVALUATOR_REGISTRY.__doc__ = """
Registry for module that is compatible with the evaluator that inherits _Metric.

The registered object will be called with `obj(cfg)`.
"""


class _Metric:
    def __init__(self, cfg):
        self.config = cfg
        self.ignore_label = cfg.DATA.ignore_label
        self._name = self.__class__.__name__

    def update(self, pred: torch.Tensor, gt: torch.Tensor, batch_offsets: torch.Tensor):
        raise NotImplementedError

    def get_value(self) -> dict:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Acc(_Metric):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.meter = AverageMeter()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor, batch_offsets: torch.Tensor):
        # the number of points for each data is different
        # so, each data in one batch should be calculated independently
        _acc_list = self.batch_offsets_accuracy(pred, gt, batch_offsets, self.ignore_label)
        if self.config.is_distributed:
            _acc = sum(_acc_list) / len(_acc_list) if len(_acc_list) > 0 else 0.0
            _avg_acc = torch.tensor(_acc).to(gt.device)
            gather_list = [torch.zeros(1).to(gt.device) for _ in range(self.config.world_size)]
            dist.all_gather(gather_list, _avg_acc)
            acc_list = [tensor.item() for tensor in gather_list]
        else:
            acc_list = _acc_list
        self.meter.update_list(acc_list)

    def get_value(self) -> dict:
        return {self._name: self.meter.avg}

    def clear(self):
        self.meter = AverageMeter()

    def _accuracy(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds: (N, )
        # targets: (N)
        if preds.size(0) <= 0:
            return 0.0
        return (torch.sum((preds == targets).float()) / preds.size(0)).item()

    def batch_offsets_accuracy(self, preds: torch.Tensor, targets: torch.Tensor, batch_offsets, ignore_label):
        # since all point are collected in the batch dimension
        # the batch offsets are used to denote the segment of each points
        # for example, if bs=2, 1000 points for points#1, 123 points for points#2
        # then the batch_offsets is [0, 1000, 1123]
        logger = logging.getLogger(PROJECT_NAME)

        acc_list = []
        for i in range(batch_offsets.size(0) - 1):
            start, end = batch_offsets[i], batch_offsets[i + 1]

            one_preds, one_targets = preds[start:end], targets[start:end]

            # we need to ignore invalid label
            valid_mask = one_targets != ignore_label

            acc_list.append(
                self._accuracy(one_preds[valid_mask], one_targets[valid_mask])
            )
        return acc_list


@EVALUATOR_REGISTRY.register()
class IoU(_Metric):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.name_2_id = CLASS2Id
        self._class_index = list(CLASS2Id.values())
        self.size = len(self._class_index)
        self.unknown = max(self._class_index) + 1
        self.iou = torch.zeros(1)
        self.iou_res = 0
        self.avg_iou = 0
        self.confusion_matrix = torch.zeros((self.size + 1, self.size + 1), dtype=torch.float32)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor, batch_offsets: torch.Tensor):
        # using confusion matrix to calculate IoU
        # pred: (N, )
        # gt: (N)
        confusion_matrix = torch.zeros((self.size + 1, self.size + 1), dtype=torch.float32).to(gt.device)
        confusion_matrix += self.count_confusion_matrix(pred, gt)

        if self.config.is_distributed:
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)

        self.confusion_matrix += confusion_matrix.cpu()

        self.iou = self.cal_iou()
        self.iou_res = {key: self.iou[i].item() for i, key in enumerate(self.name_2_id)}

    def get_value(self) -> dict:
        return {self._name: self.get_avg_iou()}

    def clear(self):
        self.confusion_matrix = torch.zeros((self.size + 1, self.size + 1), dtype=torch.float32)

    def count_confusion_matrix(self, pred: torch.Tensor, gt: torch.Tensor):
        # pred.shape: (N, )
        # gt.shape:   (N, )

        n = pred.shape[0]
        sample_weight = torch.ones(n, dtype=torch.float32).to(gt.device)

        """ 
            for labels which are not in class label, we should ignore it
            e.g. '-100' is not in class label 
        """
        # torch.where(condition) return tuple, only the first element is needed
        indices = torch.where(gt == self.ignore_label)[0].to(gt.device)
        # Because the next line will change the values in gt, we should clone gt
        _gt = gt.clone().detach()
        _gt[indices] = self.unknown

        sample_weight[indices] = 0

        indices = torch.cat([_gt, pred]).reshape(2, _gt.shape[0])
        confusion_matrix = torch.sparse_coo_tensor(indices, sample_weight, (self.size + 1, self.size + 1),
                                                   dtype=torch.float32, device=gt.device)

        return confusion_matrix

    def cal_iou(self):
        tp = torch.diagonal(self.confusion_matrix)  # true positive
        fn = torch.sum(self.confusion_matrix, dim=1)  # false negative
        fp = torch.sum(self.confusion_matrix, dim=0)  # false positive
        iou_denominator = fn + fp - tp

        # initialize output tensor with desired value
        iou = torch.full_like(tp, fill_value=float('nan'), dtype=torch.float32)
        # zero mask
        mask = (iou_denominator != 0)
        # finally perform division
        iou[mask] = (tp[mask].float() / iou_denominator[mask].float())

        return iou

    def get_classwise_iou(self):
        return self.iou_res

    def get_avg_iou(self):
        mask = (~torch.isnan(self.iou))
        self.avg_iou = torch.sum(self.iou[mask]) / self.size
        return self.avg_iou.item()

    def _load_file(self, file_path: str):
        with open(file_path, 'r') as f:
            iou_res = torch.tensor(f.read().splitlines(), dtype=torch.int64)
        return iou_res

    def load_file(self, pred_file: str, gt_file: str):
        pred, gt = self._load_file(pred_file), self._load_file(gt_file)

        print(pred_file[-16:-4])
        assert pred.shape == gt.shape, "number of predicted values does not match number of vertices"

        return pred, gt

    def _load_directory(self, dir_path: str):
        file_list = [f for f in os.listdir(dir_path)]
        temp = []
        for filename in file_list:
            with open(os.path.join(dir_path, filename), 'r') as f:
                temp += f.read().splitlines()
        _list = [int(_) for _ in temp]
        return torch.tensor(_list, dtype=torch.int64)

    def load_directory(self, pred_path: str, gt_path: str):
        pred, gt = self._load_directory(pred_path), self._load_directory(gt_path)
        assert pred.shape == gt.shape, "number of predicted values does not match number of vertices"
        return pred, gt

    def write_result_file(self, filename):
        with open(filename, 'w') as f:
            f.write('iou scores\n')
            for i, key in enumerate(self.name_2_id.keys()):
                label_id = self.name_2_id[key]
                label_name = key
                iou = self.iou_res[key]
                f.write('{0:<14s}({1:<2d}): {2:>5.3f}\n'.format(label_name, label_id, iou))
            f.write('\nconfusion matrix\n')
            f.write('\t\t\t')

            length = len(self._class_index)
            for i in range(length):
                f.write('{0:<8d}'.format(self._class_index[i]))
            f.write('\n')
            for r, key in enumerate(self.name_2_id.keys()):
                f.write('{0:<14s}({1:<2d})'.format(key, self._class_index[r]))
                for c in range(length):
                    f.write('\t{0:>5.3f}'.format(self.confusion_matrix[self._class_index[r], self._class_index[c]]))
                f.write('\n')


@EVALUATOR_REGISTRY.register()
class Accv2(_Metric):
    def update(self, pred: torch.Tensor, gt: torch.Tensor, batch_offsets: torch.Tensor):
        acc = self.precision_at_1(pred, gt).to(gt.device)
        if self.config.is_distributed:
            gather_list = [torch.zeros(1).to(gt.device) for _ in range(self.config.world_size)]
            dist.all_gather(gather_list, acc)
            acc_list = [_.item() for _ in gather_list]
            self.meter.update_list(acc_list)
        else:
            self.meter.update(acc.item())

    def get_value(self) -> dict:
        return {self._name: self.meter.avg}

    def clear(self):
        self.meter.reset()

    def __init__(self, cfg):
        super().__init__(cfg)
        self.meter = AverageMeter()

    def precision_at_1(self, pred, gt):
        """Computes the precision@k for the specified values of k"""
        # batch_size = target.size(0) * target.size(1) * target.size(2)
        mask = gt != self.ignore_label
        pred = pred[mask]
        gt = gt[mask]
        correct = pred.eq(gt)
        correct = correct.view(-1)
        if correct.nelement():
            return torch.tensor(correct.float().sum(0).mul(100.0 / correct.size(0)).item())
        else:
            return torch.tensor(0.0)


@EVALUATOR_REGISTRY.register()
class IoUv2(_Metric):
    NUM_CLASS_2_CATEGORIES = {
        13: ['clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
             'door', 'floor', 'sofa', 'table', 'wall', 'window'],
        20: ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
             'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
             'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'],
        19: ['car', 'bicycle', 'motorcycle', 'truck', 'othervehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
             'parking', 'sidewalk', 'otherground', 'building', 'fence', 'vegetation',
             'trunk', 'terrain', 'pole', 'trafficsign'],
    }

    def get_value(self) -> dict:
        # return {self._name: self.get_mIoU()}
        return self.get_mIoU_v2()

    def clear(self):
        self.confusion_matrix.zero_()

    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_class = cfg.EVALUATOR.iou_num_class

        assert self.num_class in self.NUM_CLASS_2_CATEGORIES, 'found invalid num_classes: {}'.format(self.num_class)

        self.confusion_matrix = torch.zeros((self.num_class, self.num_class), dtype=torch.float32)

        self.categories = self.NUM_CLASS_2_CATEGORIES[self.num_class]

        logging.info("Using num_class={} for calculating IoUv2".format(self.num_class))

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor, batch_offsets: torch.Tensor):
        # using confusion matrix to calculate IoU
        # pred: (N, )
        # gt: (N)
        confusion_matrix = self.fast_hist(pred, gt).to(gt.device)

        if self.config.is_distributed:
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)

        self.confusion_matrix += confusion_matrix.cpu()

    def fast_hist(self, pred: torch.Tensor, gt: torch.Tensor):
        mask = gt != self.ignore_label
        pred = pred[mask].cpu().numpy()
        gt = gt[mask].cpu().numpy()
        n = self.num_class
        np_hist = np.bincount(n * gt.astype(int) + pred, minlength=n ** 2).reshape(n, n)
        return torch.tensor(np_hist)

    def get_iou(self):
        hist = self.confusion_matrix.cpu().numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def get_mIoU(self):
        iou = self.get_iou()
        # logging.info("===> iou: {}".format(iou))
        # logging per-class iou
        x = PrettyTable()
        x.align = 'l'
        x.field_names = ['Class Id', 'IoU']
        for i, _iou in enumerate(iou):
            x.add_row([i, _iou])
        logging.debug("Per-class IoU is as followed: \n{}".format(x))
        return np.nanmean(iou)

    def get_mIoU_v2(self):
        iou = self.get_iou()
        # logging.info("===> iou: {}".format(iou))
        # logging per-class iou
        x = PrettyTable()
        x.align = 'l'
        x.field_names = ['Class Id', 'IoU']
        ret = {}
        for i, _iou in enumerate(iou):
            x.add_row([i, _iou])
            ret['Class_{}_{}_{}'.format(i, self.categories[i], self._name)] = _iou
        logging.debug("Per-class IoU is as followed: \n{}".format(x))
        ret[self._name] = np.nanmean(iou)
        return ret


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Parser for specifying WeaklySegmentationKit (WSK) config path')

    # Dataset
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to the predict directory.')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Path to the ground truth directory.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the IoU file.')

    return parser.parse_args()


# function main is for test
def main():
    config = get_arguments()

    pred_path = config.pred_path
    gt_path = config.gt_path
    output_file = config.output_file

    evaluate = IoU(CLASS2Id, -100)

    print("============ start =============")
    start_time = time.time()
    pred, gt = evaluate.load_directory(pred_path, gt_path)
    load_time = time.time()
    print("load time: ", load_time - start_time)

    evaluate.cal_iou(pred.cuda(), gt.cuda())
    print("cal time: ", time.time() - load_time)
    print("eval time: ", time.time() - start_time)

    print('\nclass-wise IoU result: ')
    for key, value in evaluate.get_classwise_iou().items():
        print(key, ": ", value)
    print('\nmean IoU:')
    print(evaluate.get_avg_iou())

    evaluate.write_result_file(output_file)


if __name__ == '__main__':
    # main()
    def run_table_string():
        x = PrettyTable()

        x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
        x.align = 'l'

        x.add_row(["Adelaide", 1295, 1158259, 600.5])
        x.add_row(["Brisbane", 5905, 1857594, 1146.4])
        x.add_row(["Darwin", 112, 120900, 1714.7])
        x.add_row(["Hobart", 1357, 205556, 6155555555555555559.5])
        x.add_row(["Sydney", 2058, 4336374, 1214.8])
        x.add_row(["Melbourne", 1566, 3806092, 646.9])
        x.add_row(["Perth", 5386, 1554769, 869.4])


    run_table_string()
