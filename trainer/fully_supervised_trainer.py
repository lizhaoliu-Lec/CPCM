import logging
import os
import pprint
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from meta_data.constant import PROJECT_NAME
from model.build import torch_intersect
from trainer.base import BaseTrainer
from util.distributed import reduce_tensor
from util.eval import calculate_remain_time
from util.utils import AverageMeter, AverageMeterDict
import MinkowskiEngine as ME
from .build import TRAINER_REGISTRY
from .lr_scheduler import build_lr_scheduler_v2
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from util.me_compatible import IS_OLD_ME


def build_optimizer_v2(cfg, model):
    name = cfg.OPTIMIZER.name
    assert name in ['SGD', 'Adam']

    # retrieve optimizer config
    config = cfg.OPTIMIZER

    # retrieve model trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())

    log_info = "Using optimizer {}, with parameters: lr={}, weight_decay={}, ".format(name, config.lr,
                                                                                      config.weight_decay)

    if name == 'SGD':
        log_info += "momentum={}, dampening={}".format(config.sgd_momentum, config.sgd_dampening)
        optimizer = SGD(
            params,
            lr=config.lr,
            momentum=config.sgd_momentum,
            dampening=config.sgd_dampening,
            weight_decay=config.weight_decay)
    elif name == 'Adam':
        log_info += "adam_beta1={}, adam_beta2={}".format(config.adam_beta1, config.adam_beta2)
        optimizer = Adam(
            params,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay)
    else:
        logging.error('Optimizer type not supported')
        raise ValueError('Optimizer type not supported')

    logging.info(log_info)

    return optimizer


def get_lower_case_name(text):
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            lst.append("_")
        lst.append(char)

    return "".join(lst).lower()


@TRAINER_REGISTRY.register()
class FullySupervisedTrainer(BaseTrainer):
    LOSS_KEYS = ['Loss', 'LossSeg']

    def __init__(self, config):

        # check has turned on the empty cuda cache for me 0.5.4 or not
        if not IS_OLD_ME and config.TRAINER.empty_cache_every < 0:
            logging.warning(
                "Empty cuda cache is not turn on for me version > 0.4.3, out of cuda memory error may occur")

        self.device = torch.cuda.current_device()
        self.cache_step_logs = []
        self.cache_epoch_logs = []  # format {'type': 'info' or 'debug', 'value': str}
        self.cache_step_tensorboard = []  # format {'type': 'scalar', 'key': k, 'value': v, 'step': t}
        self.cache_epoch_tensorboard = []

        super().__init__(config)

        self.apply_scene_cls = self.config.MODEL.apply_scene_cls
        if self.apply_scene_cls:
            self.LOSS_KEYS.append('LossScene')

    def empty_all_cache(self):
        self.cache_step_logs = []
        self.cache_epoch_logs = []
        self.cache_step_tensorboard = []
        self.cache_epoch_tensorboard = []

    def empty_step_cache(self):
        self.cache_step_logs = []
        self.cache_step_tensorboard = []

    def empty_epoch_cache(self):
        self.cache_epoch_logs = []
        self.cache_epoch_tensorboard = []

    def _log(self, info_dicts, log_type):
        TYPE2LOG = {
            'info': logging.info,
            'warn': logging.warning,
            'debug': logging.debug,
            'error': logging.error,
        }
        final_v = []
        remained_dicts = []

        for info_dict in info_dicts:
            _type = info_dict['type']
            v = info_dict['value']
            if _type == log_type:
                final_v.append(v)
            else:
                remained_dicts.append(info_dict)

        if len(final_v) > 0:
            TYPE2LOG[log_type](', '.join(final_v))

        return remained_dicts

    def log_step(self, log_type):
        self.cache_step_logs = self._log(self.cache_step_logs, log_type)

    def log_epoch(self, log_type):
        self.cache_epoch_logs = self._log(self.cache_epoch_logs, log_type)

    def _tensorboard(self, info_dicts):
        for info_dict in info_dicts:
            _type = info_dict['type']
            if _type == 'scalar':
                k, v, t = info_dict['key'], info_dict['value'], info_dict['step']
                self.writer.add_scalar(k, v, t)

    def tensorboard_step(self):
        self._tensorboard(self.cache_step_tensorboard)
        self.cache_step_tensorboard = []

    def tensorboard_epoch(self):
        self._tensorboard(self.cache_epoch_tensorboard)
        self.cache_epoch_tensorboard = []

    def is_step_to_log_for_train(self):
        return self.global_step % self.config.TRAINER.log_every == 0 or self.global_step % len(
            self.dataset.train_dataloader) == 0

    def is_step_to_log_for_val(self):
        return self.global_step % self.config.TRAINER.log_every == 0 or self.global_step % len(
            self.dataset.val_dataloader) == 0

    def is_step_to_log_for_test(self):
        return self.global_step % self.config.TRAINER.log_every == 0 or self.global_step % len(
            self.dataset.test_dataloader) == 0

    def handle_after_dataset(self):
        d_len = len(self.dataset.train_dataloader)
        num_epoch = self.config.TRAINER.epochs
        max_iter = d_len * num_epoch
        self.config.SCHEDULER.max_iter = max_iter
        logging.info("Configurate max_iter={} by d_len={} * num_epoch={}".format(
            max_iter, d_len, num_epoch
        ))

    def to_cuda_if_use(self, data):
        # .cuda() results in error for ME.SparseTensor
        if data is None:
            return data
        if not IS_OLD_ME and isinstance(data, ME.SparseTensor):
            return ME.SparseTensor(data.F, data.C, device=self.device)
        return data.to(self.device) if self.use_cuda else data

    @staticmethod
    def get_batch_offsets(coords):
        batch_offsets = [0]
        for batch_idx in torch.unique(coords[:, 0]):
            num_points = torch.sum((coords[:, 0] == batch_idx).int()).item()
            batch_offsets.append(batch_offsets[-1] + num_points)
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
        return batch_offsets

    def build_optimizer(self):
        return build_optimizer_v2(self.config, self.model)

    def build_lr_scheduler(self):
        return build_lr_scheduler_v2(self.config, self.optimizer)

    def handle_distributed_parallel(self, model):
        config = self.config

        if self.use_cuda:
            logging.info('Using cuda: {} to accelerate model training'.format(config.rank))
            model = model.to(config.rank)

        # TODO handle norm sync bn for me.bn

        if config.is_distributed:
            logging.info('Converting model to DDP model')
            model = DDP(
                module=model, device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False
            )
            logging.info('Done Converting model to DDP model')
            # DDP(model, [config.rank])
        else:
            model = model

        return model

    def step(self, batch):

        coords = batch['coords']
        feats = batch['feats']
        targets = None
        if 'labels' in batch:
            targets = self.to_cuda_if_use(batch['labels'])

        if self.config.AUGMENTATION.normalize_color:
            assert self.config.AUGMENTATION.normalize_coord is False
            feats[:, :3] = feats[:, :3] / 255. - 0.5
        if self.config.AUGMENTATION.normalize_coord:  # for semantic kitti
            assert self.config.AUGMENTATION.normalize_color is False
            feats[:, 0] = (feats[:, 0] - (-0.3527016)) / 14.5789787
            feats[:, 1] = (feats[:, 1] - 0.7280641) / 9.84908962
            feats[:, 2] = (feats[:, 2] - (-0.96655365)) / 0.90581832

        sparse_input = self.to_cuda_if_use(ME.SparseTensor(feats, coords))
        ret = self.model(sparse_input, targets)

        step_ret = {}

        if self.model.training:
            loss_seg = ret['loss']
            loss_scene = torch.zeros(1, device=self.device).float()
            if self.apply_scene_cls:
                loss_scene = ret['loss_scene']

            loss = loss_seg + loss_scene

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # wait for all the gradient to be calculated
            torch.cuda.synchronize()

            self.optimizer.step()

            if self.config.is_distributed:
                loss = reduce_tensor(loss.data, self.config.world_size)
                loss_seg = reduce_tensor(loss_seg.data, self.config.world_size)
                loss_scene = reduce_tensor(loss_scene.data, self.config.world_size)

            step_ret[self.LOSS_KEYS[0]] = loss.item()
            step_ret[self.LOSS_KEYS[1]] = loss_seg.item()
            if self.apply_scene_cls:
                step_ret[self.LOSS_KEYS[2]] = loss_scene.item()

        semantic_scores = ret['semantic_scores']
        pred = torch.argmax(semantic_scores, dim=1)

        # # log semantic_scores here
        # if self.is_step_to_log_for_train():
        #     self.cache_step_logs.append({
        #         "type": "info", "value": "semantic_scores: {}".format(semantic_scores[0])
        #     })

        step_ret['semantic_scores'] = ret['semantic_scores']
        step_ret['feats'] = ret['semantic_feats']
        step_ret['pred'] = pred

        return step_ret

    def train_one_epoch(self):
        trainer_config = self.config.TRAINER

        iter_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        loss_meters = AverageMeterDict(self.LOSS_KEYS)
        self.evaluator.clear_all()

        self.model.train()

        iter_start_time = time.time()
        data_start_time = time.time()

        dataloader_len = len(self.dataset.train_dataloader)

        for i, batch in enumerate(self.dataset.train_dataloader):
            # update data meter
            data_time_meter.update(time.time() - data_start_time)
            # empty cuda cache
            empty_cache_every = self.config.TRAINER.empty_cache_every
            if empty_cache_every > 0 and i % empty_cache_every == 0:
                # logging.info("Emptying cuda cache...")
                torch.cuda.empty_cache()
            # adjust learning rate
            self.lr_scheduler.step(epoch=self.global_step, step_type='step')

            # log out before step, which causes out of memory error
            # debug_info = "[TRAIN] epoch: {}/{}, iter: {}/{}, num_points: {}".format(
            #     self.epoch + 1, trainer_config.epochs, i + 1, dataloader_len, batch['labels'].size(0)
            # )
            self.cache_step_logs.append({
                'type': 'debug', 'value': "[TRAIN] epoch: {}/{}".format(self.epoch + 1, trainer_config.epochs)
            })
            self.cache_step_logs.append({
                'type': 'debug', 'value': "iter: {}/{}".format(i + 1, dataloader_len)
            })
            self.cache_epoch_logs.append({"type": "debug", 'value': "num_points: {}".format(batch['labels'].size(0))})
            self.log_step(log_type='debug')

            # forward the data
            if self.is_step_to_log_for_train():
                self.cache_step_logs.append({
                    'type': 'info', 'value': "[TRAIN] epoch: {}/{}, iter: {}/{}, num_points: {}".format(
                        self.epoch + 1, trainer_config.epochs, i + 1, dataloader_len, batch['labels'].size(0))
                })

            step_ret = self.step(batch)

            # update evaluator
            labels = self.to_cuda_if_use(batch['labels'])
            batch_offsets = self.to_cuda_if_use(self.get_batch_offsets(batch['coords']))
            self.evaluator.update_all(step_ret['pred'], labels, batch_offsets)

            # update meters
            loss_meters.update({k: step_ret[k] for k in self.LOSS_KEYS if k in step_ret})

            # calculate remain time
            iter_time_meter.update(time.time() - iter_start_time)
            remain_time = calculate_remain_time(
                cur_step=i + 1,
                total_step=dataloader_len,
                avg_time=iter_time_meter.avg,
            )

            if self.is_step_to_log_for_train():

                time_str = "data time: {:.2f}, iter time: {:.2f}, remain time: {}".format(
                    data_time_meter.avg, iter_time_meter.avg, remain_time
                )

                loss_str = ', '.join(
                    ["{}: {:.4f}".format(_, loss_meters[_].avg) for _ in self.LOSS_KEYS if _ in loss_meters])

                evaluator_values = self.evaluator.get_all_value()
                metrics_str = ', '.join(["{}: {:.2f}".format(k, evaluator_values[k]) for k in evaluator_values])

                self.cache_step_logs.append({
                    'type': 'info', 'value': time_str
                })
                self.cache_step_logs.append({
                    'type': 'info', 'value': loss_str
                })
                self.cache_step_logs.append({
                    'type': 'info', 'value': metrics_str
                })

                # do log
                self.log_step(log_type='info')

                # log num_points, loss_step to tensorboard
                self.cache_step_tensorboard.append({
                    'type': 'scalar',
                    'key': 'Train/NumPoints',
                    'value': labels.size(0),
                    'step': self.global_step
                })

                # log loss
                for k in self.LOSS_KEYS:
                    if k not in step_ret:
                        continue
                    self.cache_step_tensorboard.append({
                        'type': 'scalar',
                        'key': 'Train/{}Step'.format(k),
                        'value': step_ret[k],
                        'step': self.global_step
                    })
                self.tensorboard_step()

            self.global_step += 1

            # update data time
            data_start_time = time.time()

            # update iter time
            iter_start_time = time.time()

        self.cache_epoch_logs.append({
            'type': 'info', 'value': "[TRAIN] epoch: {}/{}".format(self.epoch + 1, trainer_config.epochs)
        })
        for lr_idx, param_group in enumerate(self.optimizer.param_groups):
            _lr = param_group['lr']
            self.cache_epoch_logs.append({
                'type': 'info', 'value': "Learning Rate: lr_{}: {}".format(lr_idx, _lr)
            })
            self.cache_epoch_tensorboard.append({
                'type': 'scalar',
                'key': 'Train/LR_{}'.format(lr_idx),
                'value': _lr,
                'step': self.epoch + 1
            })

        # finally, log to tensorboard
        for k in self.LOSS_KEYS:
            if k not in loss_meters:
                continue
            self.cache_epoch_tensorboard.append({
                'type': 'scalar',
                'key': 'Train/{}'.format(k),
                'value': loss_meters[k].avg,
                'step': self.epoch + 1
            })

        evaluator_value_dict = self.evaluator.get_all_value()
        for metrics_name in evaluator_value_dict:
            self.cache_epoch_tensorboard.append({
                'type': 'scalar',
                'key': f'Train/{metrics_name}',
                'value': evaluator_value_dict[metrics_name],
                'step': self.epoch + 1
            })

        self.handle_after_one_epoch()  # used to gather other info to log and tensorboard

        self.log_epoch(log_type='info')
        self.tensorboard_epoch()

    def val(self):
        logger = logging.getLogger(PROJECT_NAME)

        trainer_config = self.config.TRAINER

        iter_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        self.evaluator.clear_all()

        self.model.eval()

        iter_start_time = time.time()
        data_start_time = time.time()

        dataloader_len = len(self.dataset.val_dataloader)

        for i, batch in enumerate(self.dataset.val_dataloader):
            # update data meter
            data_time_meter.update(time.time() - data_start_time)
            # empty cuda cache
            # torch.cuda.empty_cache()

            # forward the data
            step_ret = self.step(batch)

            # update meters
            labels = self.to_cuda_if_use(batch['labels'])
            batch_offsets = self.to_cuda_if_use(self.get_batch_offsets(batch['coords']))
            self.evaluator.update_all(step_ret['pred'], labels, batch_offsets)

            # calculate remain time
            iter_time_meter.update(time.time() - iter_start_time)
            remain_time = calculate_remain_time(
                cur_step=i + 1,
                total_step=dataloader_len,
                avg_time=iter_time_meter.avg,
            )

            if i % trainer_config.log_every == 0 or i == dataloader_len - 1:
                # concatenate strings
                time_str = "data time: {:.2f}, iter time: {:.2f}, remain time: {}".format(
                    data_time_meter.avg, iter_time_meter.avg, remain_time
                )

                evaluator_value_dict = self.evaluator.get_all_value()
                metrics_str = ', '.join(["{}: {:.2f}".format(k, evaluator_value_dict[k]) for k in evaluator_value_dict])
                log_info = "[VAL] epoch: {}/{}, iter: {}/{}, {}, {}".format(
                    self.epoch + 1, trainer_config.epochs, i + 1, dataloader_len, metrics_str, time_str
                )

                # do log
                logger.info(log_info)

            # update data time
            data_start_time = time.time()

            # update iter time
            iter_start_time = time.time()

        # log to tensorboard
        metric_dict = self.evaluator.get_all_value()
        for metric_name in metric_dict:
            self.writer.add_scalar('Val/{}'.format(metric_name), metric_dict[metric_name], self.epoch + 1)

        # finally, save ckpt
        is_best = False
        # the first metrics is the most important metrics
        key_metric = self.config.EVALUATOR.metrics[0]
        if key_metric not in self.metric_dict or max(self.metric_dict[key_metric]) < metric_dict[key_metric]:
            is_best = True

        if key_metric not in self.metric_dict:
            self.metric_dict[key_metric] = []

        self.metric_dict[key_metric].append(metric_dict[key_metric])

        self.save_checkpoint(metric_dict=self.metric_dict,
                             epoch=self.epoch,
                             is_best=is_best)

    def test(self):
        pass

    def test_for_scannet(self):
        self.model.eval()
        iter_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        iter_start_time = time.time()
        data_start_time = time.time()

        dataloader_len = len(self.dataset.test_dataloader)

        for i, batch in enumerate(self.dataset.test_dataloader):
            output_path = self.config.test.scannet_testset_output_result_path
            num_points = batch['coords'].size(0)

            index = batch['indexes'].data
            coords, feats = batch['coords'].cpu(), batch['feats'].cpu()
            inverse_map = batch['inverse_map']

            test_dataset = self.dataset.test_dataset
            file_path = test_dataset.data_root / test_dataset.data_paths[index]
            scene_name = str(file_path).split('/')[-1].split('.')[0]

            # update data meter
            data_time_meter.update(time.time() - data_start_time)

            self.cache_step_logs.append({
                'type': 'debug', 'value': "[TEST] iter: {}/{}".format(i + 1, dataloader_len)
            })
            self.log_step(log_type='debug')

            # forward the data
            self.cache_step_logs.append({
                'type': 'info', 'value': "[TEST] iter: {}/{}, num_points: {}".format(
                    i + 1, dataloader_len, batch['coords'].size(0))
            })

            step_ret = self.step(batch)
            pred = step_ret['pred']

            os.makedirs(os.path.join(output_path, "semantic_txt"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "coords_colors_pred"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "coords_colors_pred_inverse"), exist_ok=True)

            with open(os.path.join(output_path, "semantic_txt", scene_name + '.txt'), 'w') as f:
                for item in pred[inverse_map]:
                    semantic_prediction = item.item()
                    # remapper label id to nyu40id
                    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                                    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
                    VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
                    remapper = {
                        0: 1,  # wall
                        1: 2,  # floor
                        2: 3,  # cabinet
                        3: 4,  # bed
                        4: 5,  # chair
                        5: 6,  # sofa
                        6: 7,  # table
                        7: 8,  # door
                        8: 9,  # window
                        9: 10,  # bookshelf
                        10: 11,  # picture
                        11: 12,  # counter
                        12: 14,  # desk
                        13: 16,  # curtain
                        14: 24,  # refridgerator
                        15: 28,  # shower curtain
                        16: 33,  # toilet
                        17: 34,  # sink
                        18: 36,  # bathtub
                        19: 39,  # otherfurniture
                    }
                    result = remapper[semantic_prediction]
                    f.write(str(result) + '\n')

            output_file = os.path.join(output_path, "coords_colors_pred", scene_name + '.pth')
            output_inverse_file = os.path.join(output_path, "coords_colors_pred_inverse", scene_name + '_inverse.pth')

            result = torch.cat([coords, feats, pred.cpu().unsqueeze(1)], 1)
            result_inverse = torch.cat([coords[inverse_map], feats[inverse_map], pred[inverse_map].cpu().unsqueeze(1)],
                                       1)

            torch.save(result, output_file)
            torch.save(result_inverse, output_inverse_file)

            # calculate remain time
            iter_time_meter.update(time.time() - iter_start_time)
            remain_time = calculate_remain_time(
                cur_step=i + 1,
                total_step=dataloader_len,
                avg_time=iter_time_meter.avg,
            )

            time_str = "data time: {:.2f}, iter time: {:.2f}, remain time: {}".format(
                data_time_meter.avg, iter_time_meter.avg, remain_time
            )

            self.cache_step_logs.append({
                'type': 'info', 'value': time_str
            })

            # do log
            self.log_step(log_type='info')

            # log num_points, loss_step to tensorboard
            self.cache_step_tensorboard.append({
                'type': 'scalar',
                'key': 'Test/NumPoints',
                'value': num_points,
                'step': self.global_step
            })

            self.tensorboard_step()

            # update data time
            data_start_time = time.time()

            # update iter time
            iter_start_time = time.time()

    def handle_after_one_epoch(self):
        pass


def align_two_stream_feats(feats1, feats2, idx2hasPoints1, idx2hasPoints2):
    num_ori_points1 = idx2hasPoints1.size(0)
    num_expect_points1 = torch.sum(idx2hasPoints1)
    num_actual_points1 = feats1.size(0)

    num_ori_points2 = idx2hasPoints2.size(0)
    num_expect_points2 = torch.sum(idx2hasPoints2)
    num_actual_points2 = feats2.size(0)

    assert num_ori_points1 == num_ori_points2, 'num_ori_points1: {}, num_ori_points2: {}'.format(
        num_ori_points1, num_ori_points2
    )
    assert num_actual_points1 == num_expect_points1, 'num_actual_points1: {}, num_expect_points1: {}'.format(
        num_ori_points1, num_expect_points1
    )
    assert num_actual_points2 == num_expect_points2, 'num_actual_points2: {}, num_expect_points2: {}'.format(
        num_ori_points2, num_expect_points2
    )

    # obtain non zero mask
    mask1 = torch.nonzero(idx2hasPoints1).squeeze(1)
    mask2 = torch.nonzero(idx2hasPoints2).squeeze(1)
    mask_intersect = torch.nonzero(idx2hasPoints1 * idx2hasPoints2).squeeze(1)

    # recover the full features
    full_feats1 = torch.zeros((idx2hasPoints1.size(0), feats1.size(1)), dtype=feats1.dtype, device=feats1.device)
    full_feats2 = torch.zeros((idx2hasPoints2.size(0), feats2.size(1)), dtype=feats2.dtype, device=feats2.device)
    full_feats1[mask1, :] = feats1
    full_feats2[mask2, :] = feats2

    # retrieve the matched features
    full_feats1 = full_feats1[mask_intersect, :]
    full_feats2 = full_feats2[mask_intersect, :]

    return full_feats1, full_feats2


def all_to_batch_other(all_coords, all_others, start_dim=0, no_dim=False):
    batch_ids = torch.unique(all_coords[:, 0])

    batch_scores = []  # (num_batch, num_point, num_class)

    for batch_id in batch_ids:
        batch_id = batch_id.item()

        batch_mask = all_coords[:, 0] == batch_id
        if not no_dim:
            batch_scores.append(all_others[batch_mask, start_dim:])
        else:
            batch_scores.append(all_others[batch_mask])
    return batch_scores


def align_two_stream_feats_v2(coords1, coords2, feats1, feats2, corr_indexes1, corr_indexes2,
                              return_list_feats=False,
                              targets1=None, targets2=None,
                              ignore_class=None,
                              unlabeled_features_only=False):
    if unlabeled_features_only is True:
        valid_for_selecting_features = targets1 is not None and targets2 is not None and ignore_class is not None
        assert valid_for_selecting_features, 'labels, ignore_class must be provided for selecting unlabeled data'

    batch_feats1 = all_to_batch_other(coords1, feats1)
    batch_feats2 = all_to_batch_other(coords2, feats2)
    batch_targets1 = None
    batch_targets2 = None

    if unlabeled_features_only is True:
        batch_targets1 = all_to_batch_other(coords1, targets1, no_dim=True)
        batch_targets2 = all_to_batch_other(coords2, targets2, no_dim=True)

    batch_corr_indexes1 = all_to_batch_other(corr_indexes1, corr_indexes1, start_dim=1)
    batch_corr_indexes1 = [_.squeeze(1) for _ in batch_corr_indexes1]
    batch_corr_indexes2 = all_to_batch_other(corr_indexes2, corr_indexes2, start_dim=1)
    batch_corr_indexes2 = [_.squeeze(1) for _ in batch_corr_indexes2]

    aligned_feats1, aligned_feats2 = [], []
    for idx in range(len(batch_feats1)):
        _f1, _f2 = batch_feats1[idx][batch_corr_indexes1[idx]], batch_feats2[idx][batch_corr_indexes2[idx]]
        # logging.info("===> _f1.size(): {}, _f2.size(): {}".format(_f1.size(), _f2.size()))
        assert _f1.size() == _f2.size()

        if unlabeled_features_only is True:
            _y1, _y2 = batch_targets1[idx][batch_corr_indexes1[idx]], batch_targets2[idx][batch_corr_indexes2[idx]]
            _f1, _f2 = _f1[_y1 == ignore_class], _f2[_y2 == ignore_class]

        aligned_feats1.append(_f1)
        aligned_feats2.append(_f2)

    if not return_list_feats:
        aligned_feats1 = torch.cat(aligned_feats1, dim=0)
        aligned_feats2 = torch.cat(aligned_feats2, dim=0)

    return aligned_feats1, aligned_feats2


# for extra processing the mask indices only
def align_two_stream_feats_v3(coords1, coords2, feats1, feats2, feats2_mask_indices,
                              corr_indexes1, corr_indexes2,
                              return_list_feats=False):
    assert feats2.size(0) == feats2_mask_indices.size(0)
    batch_feats1 = all_to_batch_other(coords1, feats1)
    batch_feats2 = all_to_batch_other(coords2, feats2)
    batch_feats2_mask_indices = all_to_batch_other(coords2, feats2_mask_indices, start_dim=1)

    batch_corr_indexes1 = all_to_batch_other(corr_indexes1, corr_indexes1, start_dim=1)
    batch_corr_indexes1 = [_.squeeze(1) for _ in batch_corr_indexes1]
    batch_corr_indexes2 = all_to_batch_other(corr_indexes2, corr_indexes2, start_dim=1)
    batch_corr_indexes2 = [_.squeeze(1) for _ in batch_corr_indexes2]
    batch_feats2_mask_indices = [_.squeeze(1) for _ in batch_feats2_mask_indices]

    aligned_feats1, aligned_feats2, aligned_feats2_mask_indices = [], [], []
    for idx in range(len(batch_feats1)):
        _f1, _f2 = batch_feats1[idx][batch_corr_indexes1[idx]], batch_feats2[idx][batch_corr_indexes2[idx]]
        _f2_mask_indices = batch_feats2_mask_indices[idx][batch_corr_indexes2[idx]]
        # logging.info("===> _f1.size(): {}, _f2.size(): {}".format(_f1.size(), _f2.size()))
        assert _f1.size() == _f2.size()
        aligned_feats1.append(_f1)
        aligned_feats2.append(_f2)
        aligned_feats2_mask_indices.append(_f2_mask_indices)

    if not return_list_feats:
        aligned_feats1 = torch.cat(aligned_feats1, dim=0)
        aligned_feats2 = torch.cat(aligned_feats2, dim=0)
        aligned_feats2_mask_indices = torch.cat(aligned_feats2_mask_indices, dim=0)

    return aligned_feats1, aligned_feats2, aligned_feats2_mask_indices


def get_start_and_end(unique_vector):
    num_ele = unique_vector.size(0)

    # randomly select two elements, then sort them
    rand_indexes = torch.randperm(num_ele, device=unique_vector.device)[:2]
    sorted_start_end = torch.sort(unique_vector[rand_indexes])[0]
    return sorted_start_end[0], sorted_start_end[1]


def get_pivots(unique_axis, grid_size):
    num_select = grid_size + 1
    stride = math.floor(unique_axis.size(0) // grid_size)
    pivot_indexes = torch.tensor(
        [min(i * stride, unique_axis.size(0) - 1) for i in range(num_select)], device=unique_axis.device)
    pivots = unique_axis[pivot_indexes]
    return pivots


def generate_grids(x_pivots, y_pivots, z_pivots):
    # slow version
    # t1_slow = time.time()
    # grids = []
    # for x_ in range(len(x_pivots) - 1):
    #     for y_ in range(len(y_pivots) - 1):
    #         for z_ in range(len(z_pivots) - 1):
    #             grids.append([
    #                 [x_pivots[x_], y_pivots[y_], z_pivots[z_]],
    #                 [x_pivots[x_ + 1], y_pivots[y_ + 1], z_pivots[z_ + 1]],
    #             ])
    # print("===> time for slow: {} ms".format(1000 * (time.time() - t1_slow)))

    # fast version:
    # t1_fast = time.time()
    grids = [[[x1, y1, z1], [x2, y2, z2]] for x1, x2 in zip(
        x_pivots[:-1], x_pivots[1:],
    ) for y1, y2 in zip(
        y_pivots[:-1], y_pivots[1:]
    ) for z1, z2 in zip(
        z_pivots[:-1], z_pivots[1:]
    )]
    # print("===> time for fast: {} ms".format(1000 * (time.time() - t1_fast)))
    grids = torch.tensor(grids, device=x_pivots.device)
    return grids


def mask_feats_by_grids(coords, feats, mask_grids, returned_indices, grid_cnt):
    # real_coords = coords[:, 1:]  # dim 0 is batch_id
    x, y, z = coords[:, 1], coords[:, 2], coords[:, 3]
    for mask_grid in mask_grids:
        left, right = mask_grid[0], mask_grid[1]
        # print("===> left.size(): {},  right.size(): {}, coords[: 1:].size(): {}, coords.size(): {}".format(
        #     left.size(), right.size(), coords[:, 1:].size(), coords.size()
        # ))
        x1, y1, z1 = left
        x2, y2, z2 = right
        x_flag = torch.logical_and(x1 < x, x <= x2)
        y_flag = torch.logical_and(y1 < y, y <= y2)
        z_flag = torch.logical_and(z1 < z, z <= z2)
        is_inside_grid = x_flag & y_flag & z_flag

        # is_inside_grid = torch.logical_and(torch.all(real_coords > left.unsqueeze(0)),
        #                                    torch.all(real_coords <= right.unsqueeze(0)))
        # print("===> num_masked: {}".format(torch.sum(is_inside_grid.float())))
        feats[is_inside_grid, ...] = 0.
        returned_indices[is_inside_grid, 1] = grid_cnt
        grid_cnt += 1
    return feats, returned_indices, grid_cnt


def get_entropy(logits):
    # logits: pointwise logits, shape: (N, num_class)
    num_class = torch.tensor(logits.size(1)).float()

    # normalize into (0, 1), shape: (N,)
    return (-1. / torch.log(num_class)) * torch.sum(logits * torch.log(logits), dim=1)


def get_confident_entropy_mask(entropy, threshold):
    return entropy < threshold


def get_classwise_entropy_flag(logits, entropies, thres, return_thres=False):
    classwise_entropy_thres = torch.zeros(size=(logits.shape[1],),
                                          device=logits.device, dtype=logits.dtype) - 1.  # (num_class, )
    pseudo_labels = torch.argmax(logits, dim=1)  # (N, )
    for unique_label in torch.unique(pseudo_labels):
        class_entropies = entropies[pseudo_labels == unique_label]
        entropy_median = torch.median(class_entropies)
        classwise_entropy_thres[unique_label] = max(entropy_median, thres)

    classwise_entropies_thres = classwise_entropy_thres[pseudo_labels]  # (N, )

    conf_flag = get_confident_entropy_mask(entropies, classwise_entropies_thres)

    if return_thres:
        return conf_flag, classwise_entropy_thres, classwise_entropies_thres
    return conf_flag


@TRAINER_REGISTRY.register()
class TwoStreamTrainer(FullySupervisedTrainer):
    LOSS_KEYS = FullySupervisedTrainer.LOSS_KEYS + ['LossSegAux', 'LossUnp']

    def __init__(self, config):
        super(TwoStreamTrainer, self).__init__(config)

        self.use_unlabeled_data_only = self.config.TRAINER.two_stream_unlabeled_data_only
        self.loss_threshold = self.config.TRAINER.two_stream_loss_threshold

        self.mask_mode = self.config.TRAINER.two_stream_mask_mode
        self.mask_ratio = self.config.TRAINER.two_stream_mask_ratio
        self.mask_grid_size = self.config.TRAINER.two_stream_mask_grid_size
        self.mask_prob = self.config.TRAINER.two_stream_mask_prob

        self.return_list_feats = self.config.TRAINER.two_stream_return_list_feats
        self.mask_extra_stream = self.config.TRAINER.two_stream_mask_extra_stream
        self.mask_corr_loss = self.config.TRAINER.two_stream_mask_corr_loss
        self.mask_self_loss = self.config.TRAINER.two_stream_mask_self_loss
        self.mask_loss_threshold = self.config.TRAINER.two_stream_mask_loss_threshold
        self.mask_loss_entropy_threshold = self.config.TRAINER.two_stream_mask_loss_entropy_threshold
        self.mask_loss_soft_weighting = self.config.TRAINER.two_stream_mask_loss_soft_weighting

        self.chunked_masked_loss_on_masked_features_only = self.config.TRAINER.two_stream_chunked_masked_loss_on_masked_features_only
        self.masked_loss_type = self.config.TRAINER.two_stream_masked_loss_type
        self.chunked_masked_loss_threshold = self.config.TRAINER.two_stream_chunked_masked_loss_threshold

        if self.mask_extra_stream:
            self.LOSS_KEYS.append('LossMask')
            self.LOSS_KEYS.append('LossMaskSelf')
        if self.mask_ratio > 0:
            logging.info("Using mask_mode={}, mask_ratio={} for aux feats masking...".format(
                self.mask_mode, self.mask_ratio))
        if self.mask_loss_threshold > 0:
            logging.info("Using mask threshold={} for mask loss...".format(self.mask_loss_threshold))

    @staticmethod
    def is_valid_for_two_stream_loss(coords, corr_indexes, corr_indexes_aux):
        batch_indexes_from_coords = torch.unique(coords[:, 0]).tolist()
        batch_indexes_from_corr_indexes = torch.unique(corr_indexes[:, 0]).tolist()
        batch_indexes_from_corr_indexes_aux = torch.unique(corr_indexes_aux[:, 0]).tolist()

        if len(batch_indexes_from_corr_indexes) == 0 or len(batch_indexes_from_corr_indexes_aux) == 0:
            ret = False
        else:
            ret = batch_indexes_from_coords == batch_indexes_from_corr_indexes \
                  and batch_indexes_from_coords == batch_indexes_from_corr_indexes_aux

        # logging.info(
        #     "===> a: {}, b: {}, c: {}, d: {}".format(batch_indexes_from_coords, batch_indexes_from_corr_indexes,
        #                                              batch_indexes_from_corr_indexes_aux, ret))

        return ret

    def get_params_dict(self):
        ts_params = super().get_params_dict()

        trainer_config = self.config.TRAINER
        for k in trainer_config:
            if k.startswith('two_stream_'):
                ts_params[k] = trainer_config[k]

        return ts_params

    def _prepare_data(self, batch, suffix=''):
        coords = batch['coords{}'.format(suffix)]
        feats = batch['feats{}'.format(suffix)]
        targets = None
        if 'labels' in batch:
            targets = self.to_cuda_if_use(batch['labels{}'.format(suffix)])
        if self.config.AUGMENTATION.normalize_color:
            assert self.config.AUGMENTATION.normalize_coord is False
            feats[:, :3] = feats[:, :3] / 255. - 0.5
        if self.config.AUGMENTATION.normalize_coord:  # for semantic kitti
            assert self.config.AUGMENTATION.normalize_color is False
            feats[:, 0] = (feats[:, 0] - (-0.3527016)) / 14.5789787
            feats[:, 1] = (feats[:, 1] - 0.7280641) / 9.84908962
            feats[:, 2] = (feats[:, 2] - (-0.96655365)) / 0.90581832
        sparse_input = self.to_cuda_if_use(ME.SparseTensor(feats, coords))
        return sparse_input, targets, coords, feats

    def _step_two_stream(self, batch):
        # prepare data for stream 1
        sparse_input, targets, coords, feats = self._prepare_data(batch)

        # process data for stream 2, note that data for stream 2 is not available during validation/testing
        sparse_input_aux, targets_aux, coords_aux, feats_aux = None, None, None, None
        if 'coords_aux' in batch:
            sparse_input_aux, targets_aux, coords_aux, feats_aux = self._prepare_data(batch, suffix='_aux')
            if self.mask_ratio > 0 and random.random() <= self.mask_prob:
                sparse_input_aux = self.mask_feats(sparse_input_aux,
                                                   mask_mode=self.mask_mode,
                                                   mask_ratio=self.mask_ratio,
                                                   mask_grid_size=self.mask_grid_size)

        ret = self.model(sparse_feats=sparse_input, sparse_feats_aux=sparse_input_aux,
                         labels=targets, labels_aux=targets_aux,
                         global_step=self.global_step)
        semantic_scores = ret['semantic_scores']
        pred = torch.argmax(semantic_scores, dim=1)

        step_ret = {}

        if self.model.training:
            loss_seg = ret['loss']
            loss_seg_aux = ret['loss_aux']

            loss_unsupervised = torch.zeros(1, device=loss_seg.device, dtype=torch.float).squeeze()
            loss_unsupervised_weight = self.get_loss_unsupervised_weight()

            # perform unsupervised loss calculation here
            FEATS_KEY = self.config.TRAINER.two_stream_feats_key
            assert FEATS_KEY in ret
            assert '{}_aux'.format(FEATS_KEY) in ret
            k, k_aux = FEATS_KEY, '{}_aux'.format(FEATS_KEY)
            unp_feats, unp_feats_aux = ret[k], ret[k_aux]
            # align two stream feats
            corr_indexes, corr_indexes_aux = self.to_cuda_if_use(
                batch['corr_indexes_batch']), self.to_cuda_if_use(batch['corr_indexes_aux_batch'])  # get corr_indexes
            # some scene does not have corr part
            if self.is_valid_for_two_stream_loss(coords, corr_indexes, corr_indexes_aux):
                if self.config.DATA.alignment_level == 'feature':  # only align if feature alignment turned on
                    unp_feats, unp_feats_aux = align_two_stream_feats_v2(
                        coords, coords_aux,
                        unp_feats, unp_feats_aux,
                        corr_indexes, corr_indexes_aux,
                        return_list_feats=self.return_list_feats,
                        targets1=targets, targets2=targets_aux,
                        ignore_class=self.config.DATA.ignore_label,
                        unlabeled_features_only=self.use_unlabeled_data_only,
                    )

                loss_unsupervised = self.unsupervised_loss(unp_feats, unp_feats_aux)
            else:
                logging.warning(
                    "No corresponding points for some scenes in this batch, skipping unsupervised loss for it...")
                if 'mlp' in FEATS_KEY:
                    # if mlp is applied, it must be optimized

                    loss_unsupervised = 1e-12 * (torch.sum(unp_feats - torch.mean(unp_feats)) + torch.sum(
                        unp_feats_aux - torch.mean(unp_feats_aux)))

            loss = loss_seg + loss_unsupervised_weight * loss_unsupervised

            if self.config.TRAINER.two_stream_seg_both:
                loss += loss_seg_aux

            # log loss weight here
            if self.is_step_to_log_for_train():
                self.cache_step_logs.append({
                    "type": "info", "value": "unp_loss_weight: {}".format(loss_unsupervised_weight)
                })
                self.cache_step_tensorboard.append({
                    "type": "scalar",
                    "key": 'TwoStream/LossWeight',
                    "value": loss_unsupervised_weight,
                    "step": self.global_step
                })

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # wait for all the gradient to be calculated
            torch.cuda.synchronize()

            self.optimizer.step()

            if self.config.is_distributed:
                loss = reduce_tensor(loss.data, self.config.world_size)
                loss_seg = reduce_tensor(loss_seg.data, self.config.world_size)
                # reduce loss_seg_aux anyway
                loss_seg_aux = reduce_tensor(loss_seg_aux.data, self.config.world_size)
                loss_unsupervised = reduce_tensor(loss_unsupervised.data, self.config.world_size)

            step_ret[self.LOSS_KEYS[0]] = loss.item()
            step_ret[self.LOSS_KEYS[1]] = loss_seg.item()
            step_ret[self.LOSS_KEYS[2]] = loss_seg_aux.item()
            step_ret[self.LOSS_KEYS[3]] = loss_unsupervised.item()

        step_ret['semantic_scores'] = ret['semantic_scores']
        step_ret['feats'] = ret['semantic_feats']
        step_ret['pred'] = pred

        return step_ret

    def _step_two_and_mask_stream(self, batch):
        # prepare data for stream 1
        sparse_input, targets, coords, feats = self._prepare_data(batch)

        # process data for stream 2, note that data for stream 2 is not available during validation/testing
        sparse_input_aux, targets_aux, coords_aux, feats_aux = None, None, None, None
        if 'coords_aux' in batch:
            sparse_input_aux, targets_aux, coords_aux, feats_aux = self._prepare_data(batch, suffix='_aux')

        ret = self.model(sparse_feats=sparse_input, sparse_feats_aux=sparse_input_aux,
                         labels=targets, labels_aux=targets_aux,
                         global_step=self.global_step)
        semantic_scores = ret['semantic_scores']
        pred = torch.argmax(semantic_scores, dim=1)

        step_ret = {}

        if self.model.training:
            # prepare masked data for single stream
            assert self.mask_ratio > 0, 'positive mask ratio is required in extra mask stream mode'
            sparse_input_aux_masked, indices_aux_masked = self.mask_feats(sparse_input_aux,
                                                                          mask_mode=self.mask_mode,
                                                                          mask_ratio=self.mask_ratio,
                                                                          mask_grid_size=self.mask_grid_size,
                                                                          return_indices=True)
            targets_aux_masked = targets_aux.clone()
            ret_aux_masked = self.model(sparse_feats_single=sparse_input_aux_masked,
                                        sparse_labels_single=targets_aux_masked,
                                        single_stream_forward=True)  # ignore the loss here

            loss_seg = ret['loss']
            loss_seg_aux = ret['loss_aux']

            loss_unsupervised = torch.zeros(1, device=loss_seg.device, dtype=torch.float).squeeze()
            loss_unsupervised_weight = self.get_loss_unsupervised_weight()

            loss_masked = torch.zeros(1, device=loss_seg.device, dtype=torch.float).squeeze()
            loss_masked_self = torch.zeros(1, device=loss_seg.device, dtype=torch.float).squeeze()
            loss_masked_weight = self.get_loss_masked_weight()

            # perform unsupervised loss calculation here
            # (1) prepare keys and feats
            # keys
            FEATS_KEY = self.config.TRAINER.two_stream_feats_key
            FEATS_KEY_MASKED = self.config.TRAINER.two_stream_mask_feats_key
            k, k_aux = FEATS_KEY, '{}_aux'.format(FEATS_KEY)
            k_not_masked, k_not_masked_aux, k_masked_aux \
                = FEATS_KEY_MASKED, '{}_aux'.format(FEATS_KEY_MASKED), FEATS_KEY_MASKED
            assert k in ret and k_aux in ret  # unp loss check
            assert k_not_masked in ret and k_not_masked_aux in ret
            assert k_masked_aux in ret_aux_masked  # mask loss check
            # feats
            unp_feats, unp_feats_aux = ret[k], ret[k_aux]
            # use clone to avoid in place modification
            feats_not_masked, feats_aux_not_masked = ret[k_not_masked].clone(), ret[k_not_masked_aux].clone()
            feats_aux_masked = ret_aux_masked[k_masked_aux]

            # get corr indexes
            corr_indexes, corr_indexes_aux = self.to_cuda_if_use(
                batch['corr_indexes_batch']), self.to_cuda_if_use(batch['corr_indexes_aux_batch'])  # get corr_indexes
            # some scene does not have corr part
            # TODO, decouple the feats for unp loss and mask loss
            if self.is_valid_for_two_stream_loss(coords, corr_indexes, corr_indexes_aux):
                unp_feats, unp_feats_aux = align_two_stream_feats_v2(
                    coords, coords_aux,
                    unp_feats, unp_feats_aux,
                    corr_indexes, corr_indexes_aux,
                    return_list_feats=self.return_list_feats,
                )
                loss_unsupervised = self.unsupervised_loss(unp_feats, unp_feats_aux)

                # detach target here
                # Note: clone before detach

                # calculate self loss first, align will change the dimension of masked_feats_aux
                # TODO mask_self_loss does nor required corr part, move outside?
                if self.mask_self_loss:
                    # no align is required for self mask loss
                    loss_masked_self = self.masked_loss(feats_aux_not_masked.detach(),
                                                        feats_aux_masked,
                                                        indices_aux_masked[:, 1],  # dim 0 is batch id
                                                        suffix='self')
                if self.mask_corr_loss:
                    feats_not_masked, feats_aux_masked, indices_aux_masked = align_two_stream_feats_v3(
                        coords, coords_aux,
                        feats_not_masked, feats_aux_masked,
                        indices_aux_masked,
                        corr_indexes, corr_indexes_aux,
                        return_list_feats=self.return_list_feats,
                    )
                    loss_masked = self.masked_loss(feats_not_masked.detach(), feats_aux_masked,
                                                   indices_aux_masked, suffix='corr')

            else:
                logging.warning(
                    "No corresponding points for some scenes in this batch, skipping unsupervised loss for it...")
                if 'mlp' in FEATS_KEY:
                    # if mlp is applied, it must be optimized

                    loss_unsupervised = 1e-12 * (torch.sum(unp_feats - torch.mean(unp_feats)) + torch.sum(
                        unp_feats_aux - torch.mean(unp_feats_aux)))

            loss = loss_seg \
                   + loss_unsupervised_weight * loss_unsupervised \
                   + loss_masked_weight * (loss_masked + loss_masked_self)

            if self.config.TRAINER.two_stream_seg_both:
                loss += loss_seg_aux

            # log loss weight here
            if self.is_step_to_log_for_train():
                self.cache_step_logs.append({
                    "type": "info", "value": "unp_loss_weight: {}".format(loss_unsupervised_weight)
                })
                self.cache_step_tensorboard.append({
                    "type": "scalar",
                    "key": 'TwoStream/LossWeight',
                    "value": loss_unsupervised_weight,
                    "step": self.global_step
                })
                self.cache_step_logs.append({
                    "type": "info", "value": "mask_loss_weight: {}".format(loss_masked_weight)
                })
                self.cache_step_tensorboard.append({
                    "type": "scalar",
                    "key": 'TwoStream/LossMaskWeight',
                    "value": loss_masked_weight,
                    "step": self.global_step
                })

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # wait for all the gradient to be calculated
            torch.cuda.synchronize()

            self.optimizer.step()

            if self.config.is_distributed:
                loss = reduce_tensor(loss.data, self.config.world_size)
                loss_seg = reduce_tensor(loss_seg.data, self.config.world_size)
                # reduce loss_seg_aux anyway
                loss_seg_aux = reduce_tensor(loss_seg_aux.data, self.config.world_size)
                loss_unsupervised = reduce_tensor(loss_unsupervised.data, self.config.world_size)
                loss_masked = reduce_tensor(loss_masked.data, self.config.world_size)
                loss_masked_self = reduce_tensor(loss_masked_self.data, self.config.world_size)

            step_ret[self.LOSS_KEYS[0]] = loss.item()
            step_ret[self.LOSS_KEYS[1]] = loss_seg.item()
            step_ret[self.LOSS_KEYS[2]] = loss_seg_aux.item()
            step_ret[self.LOSS_KEYS[3]] = loss_unsupervised.item()
            step_ret[self.LOSS_KEYS[4]] = loss_masked.item()
            step_ret[self.LOSS_KEYS[5]] = loss_masked_self.item()

        step_ret['semantic_scores'] = ret['semantic_scores']
        step_ret['feats'] = ret['semantic_feats']
        step_ret['pred'] = pred

        return step_ret

    def step(self, batch):
        if not self.mask_extra_stream:
            return self._step_two_stream(batch=batch)
        else:
            return self._step_two_and_mask_stream(batch=batch)

    def get_loss_unsupervised_weight(self):
        start_epoch = self.config.TRAINER.two_stream_loss_start_epoch

        if self.epoch < start_epoch:
            return 0.0

        warmup_start_epoch = self.epoch
        if start_epoch > 0:
            warmup_start_epoch = self.epoch - start_epoch

        loss_unsupervised_weight = self.config.TRAINER.two_stream_loss_weight
        warmup_epoch = self.config.TRAINER.two_stream_loss_warmup_epoch
        return loss_unsupervised_weight * sigmoid_rampup(warmup_start_epoch, warmup_epoch)

    def get_loss_masked_weight(self):
        start_epoch = self.config.TRAINER.two_stream_mask_start_epoch

        if self.epoch < start_epoch:
            return 0.0

        warmup_start_epoch = self.epoch
        if start_epoch > 0:
            warmup_start_epoch = self.epoch - start_epoch

        loss_masked_weight = self.config.TRAINER.two_stream_loss_mask_weight
        warmup_epoch = self.config.TRAINER.two_stream_loss_mask_warmup_epoch
        return loss_masked_weight * sigmoid_rampup(warmup_start_epoch, warmup_epoch)

    def unsupervised_loss(self, feats, feats_aux, used_for_mask_loss=False, soft_weights=None):
        if not used_for_mask_loss:
            mode = self.config.TRAINER.two_stream_loss_mode
        else:
            mode = self.config.TRAINER.two_stream_loss_mask_mode

        if soft_weights is not None:
            assert mode == 'js_divergence_v2', 'soft_weight only available for js_divergence_v2'

        # add threshold selecting here
        if not used_for_mask_loss and self.loss_threshold > 0:
            with torch.no_grad():
                # TODO belong to the same class or not ?
                logits, logits_aux = torch.softmax(feats, dim=1), torch.softmax(feats_aux, dim=1)
                max_logits, max_logits_aux = torch.max(logits, dim=1)[0], torch.max(logits_aux, dim=1)[0]
                conf_flag = torch.logical_and(max_logits > self.loss_threshold,
                                              max_logits_aux > self.loss_threshold)
                num_selected = torch.sum(conf_flag.float()).item()
                percentage = num_selected / logits.size(0)
                if self.is_step_to_log_for_train():
                    self.cache_step_logs.append({
                        "type": "info", "value": "select_thres_percentage: {}".format(percentage)
                    })
                    self.cache_step_tensorboard.append({
                        "type": "scalar",
                        "key": 'TwoStream/SelectThresPercentage'.format(percentage),
                        "value": percentage,
                        "step": self.global_step
                    })
            feats, feats_aux = feats.clone()[conf_flag, :], feats_aux.clone()[conf_flag, :]

        if mode == 'mse':
            return F.mse_loss(feats, feats_aux)
        if mode == 'cosine_sim':
            pass
            # feats = F.normalize(feats, p=2, dim=1)
            # feats_aux = F.normalize(feats_aux, p=2, dim=1)
            # return torch.mean()
        # TODO, add sample_wise point info nce loss
        if mode == 'point_info_nce':
            return self.point_info_nce_loss(feats, feats_aux,
                                            num_sampled=self.config.TRAINER.two_stream_contrastive_num_sample,
                                            T=self.config.TRAINER.two_stream_contrastive_temperature)
        if mode == 'self_training':
            pass
        if mode == 'js_divergence':
            assert feats.ndim == 2
            assert feats_aux.ndim == 2
            # best perform at logits level
            p1, p2 = F.softmax(feats, dim=1), F.softmax(feats_aux, dim=1)
            # eps set ot 1e-4 according to PSD
            eps = 1e-4
            q = 0.5 * (p1 + p2)
            return torch.mean(p1 * torch.log(p1 / (q + eps) + eps) + p2 * torch.log(p2 / (q + eps) + eps))

        if mode == 'js_divergence_v2':
            assert feats.ndim == 2
            assert feats_aux.ndim == 2
            # best perform at logits level
            p1, p2 = F.softmax(feats, dim=1), F.softmax(feats_aux, dim=1)  # (N, num_cls), (N, num_cls)
            # eps set ot 1e-4 according to PSD
            eps = 1e-4
            q = 0.5 * (p1 + p2)
            p1q = p1 * torch.log(p1 / (q + eps) + eps)
            p2q = p2 * torch.log(p2 / (q + eps) + eps)
            if soft_weights is not None:  # soft_weights: (N, 1)
                raw_loss = (p1q + p2q).sum(dim=1)  # (N, 1)
                return (soft_weights * raw_loss).mean(dim=0)
            return (p1q + p2q).sum(dim=1).mean(dim=0)

        if mode == 'sample_point_info_nce':
            assert isinstance(feats, list), 'required list but got type `{}`'.format(type(feats))
            assert isinstance(feats_aux, list), 'required list but got type `{}`'.format(type(feats_aux))
            loss = 0
            num_sample_per_batch = len(feats)
            num_sampled_point = self.config.TRAINER.two_stream_contrastive_num_sample
            T = self.config.TRAINER.two_stream_contrastive_temperature
            for a, b in zip(feats, feats_aux):
                loss += self.point_info_nce_loss(a, b, num_sampled_point, T)
            return loss / num_sample_per_batch

        raise ValueError('Unsupported unsupervised loss.')

    def chunked_masked_loss(self, feats, feats_masked, chunk_indices, suffix):
        # suffix will be used
        chunked_feats, chunked_feats_masked = [], []

        selected_chunk_idxes = torch.unique(chunk_indices)
        if self.chunked_masked_loss_on_masked_features_only:  # filter not masked features
            selected_chunk_idxes = selected_chunk_idxes[selected_chunk_idxes != -1]  # -1 means has not MASKED

        with torch.no_grad():
            logits = torch.softmax(feats, dim=1)
            max_logits = torch.max(logits, dim=1)[0]

        for chunk_id in selected_chunk_idxes:
            selected_chunk_flag = chunk_indices == chunk_id
            _chunk_feats = feats[selected_chunk_flag, :]
            _chunk_feats_masked = feats_masked[selected_chunk_flag, :]

            # select grid feats by threshold
            if chunk_id >= 0 and self.chunked_masked_loss_threshold > 0:
                # select the confident feats to calculate mask loss
                # only scores are supported now
                assert 'scores' in self.config.TRAINER.two_stream_mask_feats_key, '{}'.format(
                    self.config.TRAINER.two_stream_mask_feats_key)
                avg_score = torch.mean(max_logits[selected_chunk_flag])
                if avg_score <= self.chunked_masked_loss_threshold:  # skip unconfident chunk
                    continue

            chunked_feats.append(feats[selected_chunk_flag, :])
            chunked_feats_masked.append(feats_masked[selected_chunk_flag, :])
        if self.is_step_to_log_for_train():
            self.cache_step_logs.append({
                "type": "info", "value": "chunk_{}_thres_num: {}".format(
                    suffix, len(chunked_feats))
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'TwoStream/Chunk{}ThresNum'.format(suffix.capitalize()),
                "value": len(chunked_feats),
                "step": self.global_step
            })
        chunked_feats = torch.cat(chunked_feats, dim=0)
        chunked_feats_masked = torch.cat(chunked_feats_masked, dim=0)

        # TODO for now, use unp loss
        return self.unsupervised_loss(chunked_feats, chunked_feats_masked, used_for_mask_loss=True)

    def plain_masked_loss(self, feats, masked_feats, suffix):
        if self.mask_loss_threshold > 0:
            # select the confident feats to calculate mask loss
            # only scores are supported now
            assert 'scores' in self.config.TRAINER.two_stream_mask_feats_key, '{}'.format(
                self.config.TRAINER.two_stream_mask_feats_key)
            # get mask first
            with torch.no_grad():
                logits = torch.softmax(feats, dim=1)
                max_logits = torch.max(logits, dim=1)[0]
                conf_flag = max_logits > self.mask_loss_threshold
                num_not_masked = torch.sum(conf_flag.float()).item()
                percentage = num_not_masked / logits.size(0)
                if self.is_step_to_log_for_train():
                    self.cache_step_logs.append({
                        "type": "info", "value": "mask_{}_thres_percentage: {}".format(
                            suffix, percentage)
                    })
                    self.cache_step_tensorboard.append({
                        "type": "scalar",
                        "key": 'TwoStream/Mask{}ThresPercentage'.format(suffix.capitalize()),
                        "value": percentage,
                        "step": self.global_step
                    })
            # TODO for now, use unp loss
            return self.unsupervised_loss(feats[conf_flag, :], masked_feats[conf_flag, :], used_for_mask_loss=True)

        # FIXME refactor the code
        if self.mask_loss_entropy_threshold > 0 or self.mask_loss_soft_weighting:
            # select the confident feats by entropy to calculate mask loss
            # only scores are supported now
            assert 'scores' in self.config.TRAINER.two_stream_mask_feats_key, '{}'.format(
                self.config.TRAINER.two_stream_mask_feats_key)
            # get mask first
            with torch.no_grad():
                logits = torch.softmax(feats, dim=1)
                entropies = get_entropy(logits)
                # entropies_classwise = get_classwise_entropy(logits, entropies)
                # conf_flag = get_confident_entropy_mask(entropies, self.mask_loss_entropy_threshold)
                if not self.mask_loss_soft_weighting:
                    conf_flag, classwise_thres, classwise_entropies_thres = get_classwise_entropy_flag(
                        logits, entropies, self.mask_loss_entropy_threshold, return_thres=True)
                else:
                    conf_flag, classwise_thres, classwise_entropies_thres = get_classwise_entropy_flag(
                        logits, entropies, 0.001, return_thres=True)  # set to 0.001 to reflect true entropies

                soft_weights = 1. - entropies
                # logging.info("===> classwise_entropies_thres mean: {}".format(torch.mean(classwise_entropies_thres)))
                # logging.info("===> soft_weights mean: {}".format(torch.mean(soft_weights)))

                num_not_masked = torch.sum(conf_flag.float()).item()
                percentage = num_not_masked / logits.size(0)
                if self.is_step_to_log_for_train():
                    self.cache_step_logs.append({
                        "type": "info",
                        "value": "mask_{}_entropy_thres_percentage: {}, mask_entropy_thres: {}".format(
                            suffix, percentage, self.mask_loss_entropy_threshold)
                    })
                    self.cache_step_tensorboard.append({
                        "type": "scalar",
                        "key": 'TwoStream/Mask{}EntropyThresPercentage'.format(suffix.capitalize()),
                        "value": percentage,
                        "step": self.global_step
                    })
                    self.cache_step_tensorboard.append({
                        "type": "scalar",
                        "key": 'TwoStream/MaskEntropyThres',
                        "value": self.mask_loss_entropy_threshold,
                        "step": self.global_step
                    })

                    for cls_idx in range(classwise_thres.shape[0]):
                        cls_thres = classwise_thres[cls_idx]
                        if cls_thres > 0:
                            self.cache_step_logs.append({
                                "type": "info",
                                "value": "class{}_entropy_thres_percentage: {}".format(cls_idx, cls_thres)
                            })
                            self.cache_step_tensorboard.append({
                                "type": "scalar",
                                "key": 'EntropyThres/Class{}'.format(cls_idx),
                                "value": cls_thres,
                                "step": self.global_step
                            })

            # TODO for now, use unp loss
            if self.mask_loss_soft_weighting:
                return self.unsupervised_loss(feats, masked_feats,
                                              used_for_mask_loss=True,
                                              soft_weights=soft_weights)
            else:
                return self.unsupervised_loss(feats[conf_flag, :], masked_feats[conf_flag, :],
                                              used_for_mask_loss=True)

        # TODO for now, use unp loss
        return self.unsupervised_loss(feats, masked_feats, used_for_mask_loss=True)

    def masked_loss(self, feats, feats_masked, chunk_indices, suffix):
        mask_loss_type = self.masked_loss_type

        if mask_loss_type == 'plain':
            return self.plain_masked_loss(feats, feats_masked, suffix)
        if mask_loss_type == 'chunked':
            return self.chunked_masked_loss(feats, feats_masked, chunk_indices, suffix)

        raise ValueError('Unsupported masked loss type: {}'.format(mask_loss_type))

    def get_mask_ratio(self):
        mask_ratio_mode = self.config.TRAINER.two_stream_mask_ratio_mode
        if mask_ratio_mode == 'constant':
            return self.mask_ratio
        if mask_ratio_mode == 'warmup':
            warmup_epoch = self.config.TRAINER.two_stream_mask_warmup_epoch
            return self.mask_ratio * sigmoid_rampup(self.epoch, warmup_epoch)

    @staticmethod
    def mask_feats(sparse_feats, mask_mode, mask_ratio, mask_grid_size, return_indices=False):
        all_coords, all_feats = sparse_feats.C, sparse_feats.F
        returned_indices = None
        me_feats = None
        if mask_mode == 'random':
            returned_indices = torch.ones(size=(all_feats.size(0), 2),
                                          device=all_feats.device) * -1  # dim0, batch_id, dim1, mask_grid_id
            returned_indices[:, 0] = all_feats[:, 0]

            num_points = all_feats.size(0)
            num_mask = int(num_points * mask_ratio)
            mask_indexes = torch.randperm(num_points, device=all_feats.device)[:num_mask]
            all_feats[mask_indexes, :3] = 0.
            returned_indices[mask_indexes, 1] = 0  # FIXME all within a chunk
            me_feats = ME.SparseTensor(all_feats, all_coords)

        if mask_mode == 'continuous':
            # for each sample in the batch
            batch_ids = torch.unique(all_coords[:, 0])
            batch_feat_list = []
            batch_returned_indices_list = []
            box_cnt = 0
            for batch_id in batch_ids:
                batch_mask = all_coords[:, 0] == batch_id
                batch_coord = all_coords[batch_mask, ...]
                batch_feat = all_feats[batch_mask, ...]

                num_points = batch_coord.size(0)
                num_mask = int(batch_coord.size(0) * mask_ratio)

                num_has_masked = 0
                x, y, z = batch_coord[:, 1], batch_coord[:, 2], batch_coord[:, 3]
                retain_flag = torch.ones(num_points, dtype=torch.bool, device=batch_feat.device)  # means retain or not
                unique_x = torch.unique(x)
                unique_y = torch.unique(y)
                unique_z = torch.unique(z)

                batch_returned_indices = torch.ones(size=(batch_feat.size(0), 2),
                                                    device=batch_feat.device) * -1  # dim0, batch_id, dim1, mask_grid_id
                batch_returned_indices[:, 0] = batch_coord[:, 0]

                while num_has_masked < num_mask:
                    x1, x2 = get_start_and_end(unique_x)
                    y1, y2 = get_start_and_end(unique_y)
                    z1, z2 = get_start_and_end(unique_z)
                    x_flag = torch.logical_and(x1 < x, x < x2)
                    y_flag = torch.logical_and(y1 < y, y < y2)
                    z_flag = torch.logical_and(z1 < z, z < z2)
                    valid_flag = x_flag & y_flag & z_flag
                    temp_retain_flag = retain_flag.clone()
                    temp_retain_flag[valid_flag] = False
                    num_has_masked = num_points - torch.sum(retain_flag.float()).item()
                    if num_has_masked > num_mask:  # do not mask more than mask_ratio percent points
                        break
                    retain_flag = temp_retain_flag
                    batch_returned_indices[valid_flag, 1] = box_cnt
                    box_cnt += 1
                # how to get the mask indexes?
                mask_indexes = torch.nonzero(~retain_flag, as_tuple=True)[0]
                batch_feat[mask_indexes, ...] = 0.
                batch_feat_list.append(batch_feat)
                batch_returned_indices_list.append(batch_returned_indices)

            all_feats = torch.cat(batch_feat_list, dim=0)
            returned_indices = torch.cat(batch_returned_indices_list, dim=0)

            me_feats = ME.SparseTensor(all_feats, all_coords)

        if mask_mode == 'grid':
            grid_size = mask_grid_size
            num_grid = grid_size ** 3
            num_mask_grid = int(num_grid * mask_ratio)

            # for each sample in the batch
            batch_ids = torch.unique(all_coords[:, 0])
            batch_feat_list = []
            batch_returned_indices_list = []
            grid_cnt = 0
            for batch_id in batch_ids:
                batch_mask = all_coords[:, 0] == batch_id
                batch_coord = all_coords[batch_mask, ...]
                batch_feat = all_feats[batch_mask, ...]

                x, y, z = batch_coord[:, 1], batch_coord[:, 2], batch_coord[:, 3]
                unique_x = torch.unique(x)
                unique_y = torch.unique(y)
                unique_z = torch.unique(z)

                batch_returned_indices = torch.ones(size=(batch_feat.size(0), 2),
                                                    device=batch_feat.device) * -1  # dim0, batch_id, dim1, mask_grid_id
                batch_returned_indices[:, 0] = batch_coord[:, 0]

                # get pivots
                x_pivots = get_pivots(unique_x, grid_size)
                y_pivots = get_pivots(unique_y, grid_size)
                z_pivots = get_pivots(unique_z, grid_size)

                # construct grid boundary
                grid_boundaries = generate_grids(x_pivots, y_pivots, z_pivots)

                # select grids to mask
                mask_grid_boundaries = grid_boundaries[
                    torch.randperm(num_grid, device=batch_coord.device)[:num_mask_grid], ...]

                # mask feats by grid
                batch_feat, batch_returned_indices, grid_cnt = mask_feats_by_grids(
                    batch_coord, batch_feat, mask_grid_boundaries, batch_returned_indices, grid_cnt)

                batch_feat_list.append(batch_feat)
                batch_returned_indices_list.append(batch_returned_indices)

            all_feats = torch.cat(batch_feat_list, dim=0)
            returned_indices = torch.cat(batch_returned_indices_list, dim=0)

            me_feats = ME.SparseTensor(all_feats, all_coords)

        assert me_feats is not None, 'Unsupported mask mode: {}'.format(mask_mode)

        if return_indices:
            return me_feats, returned_indices
        return me_feats

    @staticmethod
    def point_info_nce_loss(feats, feats_aux, num_sampled, T):
        n_points = feats.size(0)
        n_sampled = min(num_sampled, n_points)
        index_sampled = torch.randperm(n_points)[:n_sampled]

        q = feats[index_sampled, :]
        k = feats_aux[index_sampled, :]

        # normalize both q and k
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # logits: NxK
        logits = torch.matmul(q, k.T)

        # apply temperature
        logits /= T

        # labels: positive key indicators
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)


def euclidean_distance(coords1, coords2):
    """
    coords1: coordinates of the anchor, shape (N, 3)
    coords2: coordinates of the points, shape (M, 3)
    N << 3
    """
    return torch.cdist(coords1.float(), coords2.float(), p=2)  # (N, M)


def torch_1d_index(tensor, value):
    assert tensor.ndim == 1, 'tensor with ndim={}'.format(tensor.ndim)
    return torch.nonzero((tensor == value).int(), as_tuple=True)[0]


def torch_1d_neq_index(tensor, value):
    assert tensor.ndim == 1, 'tensor with ndim={}'.format(tensor.ndim)
    return torch.nonzero((tensor != value).int(), as_tuple=True)[0]


def torch_1d_random_select(tensor, num_select):
    N = tensor.size(0)
    assert num_select <= N
    return tensor[torch.randperm(N)[:num_select]]


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Queue(nn.Module):
    def __init__(self, num_feat, dim_feat, device):
        super().__init__()
        self.num_feat = num_feat
        self.dim_feat = dim_feat
        self.register_buffer("queue", torch.randn(num_feat, dim_feat))
        self.queue = nn.functional.normalize(self.queue, dim=1).to(device)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def get_max_division(self, bs):
        for _ in list(range(1, bs + 1))[::-1]:
            if self.num_feat % _ == 0:
                return _

    @torch.no_grad()
    def dequeue_and_enqueue_v2(self, keys):
        keys = concat_all_gather(keys)
        ptr = int(self.queue_ptr)
        cur_size = keys.shape[0]
        # two cases:
        # (1) ptr + cur_size <= self.num_feat
        # (2) ptr + cur_size > self.num_feat
        if ptr + cur_size <= self.num_feat:
            self.queue[ptr:ptr + cur_size, :] = keys
            # ptr = (ptr + cur_size) % self.num_feat
            ptr = ptr + cur_size
            if ptr == self.num_feat:
                ptr = 0
        else:
            tail_size = self.num_feat - ptr
            self.queue[-tail_size:, :] = keys[:tail_size, :]
            head_size = cur_size - tail_size
            self.queue[:head_size, :] = keys[tail_size:, :]
            ptr = head_size

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # TODO bug here
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        ptr = int(self.queue_ptr)

        batch_size = keys.shape[0]

        # truncate keys
        num_preserved = self.get_max_division(min(batch_size, self.num_feat - ptr))

        keys = keys[:num_preserved, :]
        batch_size = keys.shape[0]  # update bs

        assert self.num_feat % batch_size == 0, 'num_feat: {}, batch_size: {}'.format(self.num_feat,
                                                                                      batch_size)  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.num_feat  # move pointer

        self.queue_ptr[0] = ptr

    def get_latest(self):
        return self.queue[self.queue_ptr - 1, :]


class ClassQueue:
    def __init__(self, num_class, num_feat, dim_feat, device):
        self.num_class = num_class
        self.num_feat = num_feat
        self.dim_feat = dim_feat
        self.queues = {_: Queue(num_feat, dim_feat, device) for _ in range(num_class)}

    def dequeue_and_enqueue(self, keys, labels):
        for label in labels:
            self.queues[label.item()].dequeue_and_enqueue_v2(keys[labels == label])

    def get_neg_feats(self, label):
        return torch.cat([self.queues[_].queue for _ in range(self.num_class) if _ != label.item()], dim=0)

    def get_latest_pos_feat(self, label):
        return self.queues[label].get_latest()

    def get_batch_neg_feats(self, labels):
        # labels (N, 1)
        bs = labels.size(0)
        all_feats = torch.stack([self.queues[_].queue for _ in range(self.num_class)], dim=0)  # (n_cls, n_feat, dim)
        cls_idx = torch.stack([torch.arange(self.num_class) for _ in range(bs)], dim=0).to(labels.device)
        retrieved_index = cls_idx != labels.unsqueeze(1)
        batch_neg_feats = torch.stack([all_feats[_] for _ in retrieved_index], dim=0)
        batch_neg_feats = batch_neg_feats.view((batch_neg_feats.size(0), -1, batch_neg_feats.size(-1)))

        assert batch_neg_feats.size(1) == (self.num_class - 1) * self.num_feat

        return batch_neg_feats  # (N, n_class-1 * n_feat, dim)

    def get_batch_latest_pos_feats(self, labels):
        # labels (N, 1)
        latest_ptrs = torch.stack([self.queues[_].queue_ptr for _ in range(self.num_class)], dim=0).squeeze(
            1)  # (n_cls)
        all_feats = torch.stack([self.queues[_].queue for _ in range(self.num_class)], dim=0)  # (n_cls, n_feat, dim)
        return all_feats[labels, latest_ptrs[labels], :].squeeze(1)  # (N, dim)


def get_manifold(feats1, feats2, K=10, eta=1000):
    feat_dist = euclidean_distance(feats1, feats2)

    feat_w = torch.zeros_like(feat_dist)

    # k nearest neighbors
    feat_values, feat_indices = feat_dist.topk(K, dim=1, largest=False, sorted=True)

    for idx, feat_idx in enumerate(feat_indices):
        feat_w[idx, feat_idx] = torch.exp(-1 * feat_dist[idx, feat_idx] / eta)

    return feat_w


@TRAINER_REGISTRY.register()
class PseudoInstanceContrastiveTrainer(FullySupervisedTrainer):
    LOSS_KEYS = FullySupervisedTrainer.LOSS_KEYS + ['LossUnp']

    def __init__(self, config):
        super(PseudoInstanceContrastiveTrainer, self).__init__(config)

        self.class_queue = None
        cfg = self.config
        if 'bank' in cfg.TRAINER.pseudo_instance_contrastive_loss_mode:
            num_feat = cfg.TRAINER.pseudo_instance_bank_num_feat
            dim_feat = cfg.TRAINER.pseudo_instance_bank_dim_feat
            self.class_queue = ClassQueue(num_class=cfg.MODEL.out_channels,
                                          num_feat=num_feat, dim_feat=dim_feat,
                                          device=self.device)

    def get_params_dict(self):
        pic_params = super().get_params_dict()

        trainer_config = self.config.TRAINER
        for k in trainer_config:
            if k.startswith('pseudo_instance_'):
                pic_params[k] = trainer_config[k]

        return pic_params

    def step(self, batch):
        coords = batch['coords']
        feats = batch['feats']
        targets = None
        if 'labels' in batch:
            targets = self.to_cuda_if_use(batch['labels'])

        if self.config.AUGMENTATION.normalize_color:
            assert self.config.AUGMENTATION.normalize_coord is False
            feats[:, :3] = feats[:, :3] / 255. - 0.5
        if self.config.AUGMENTATION.normalize_coord:  # for semantic kitti
            assert self.config.AUGMENTATION.normalize_color is False
            feats[:, 0] = (feats[:, 0] - (-0.3527016)) / 14.5789787
            feats[:, 1] = (feats[:, 1] - 0.7280641) / 9.84908962
            feats[:, 2] = (feats[:, 2] - (-0.96655365)) / 0.90581832

        sparse_input = self.to_cuda_if_use(ME.SparseTensor(feats, coords))

        # produce buggy label to debug
        # buggy_labels = torch.zeros_like(targets) + self.config.DATA.ignore_label
        # buggy_labels[111] = 1
        # # buggy_labels[1110] = 2
        # # buggy_labels[2220] = 3
        # buggy_labels[1110] = 1
        # buggy_labels[2220] = 1
        # ret = self.model(sparse_input, buggy_labels)

        ret = self.model(sparse_input, targets)
        semantic_scores = ret['semantic_scores']
        pred = torch.argmax(semantic_scores, dim=1)

        step_ret = {}

        if self.model.training:
            loss_seg = ret['loss']

            # perform pseudo contrastive loss here
            FEATS_KEY = self.config.TRAINER.pseudo_instance_contrastive_feats_key
            assert FEATS_KEY in ret
            cont_feats = ret[FEATS_KEY]
            loss_contrastive_weight = self.get_loss_contrastive_weight()
            loss_contrastive = torch.zeros(1).to(self.device)

            num_pseudo_instance = 0
            if cont_feats is not None:
                if FEATS_KEY != 'pic_feats':
                    pseudo_instances, cont_labels = self.get_pseudo_instance(
                        coords, feats, cont_feats, targets,
                        predicted_scores=torch.softmax(semantic_scores, dim=1))
                else:
                    pseudo_instances, cont_labels = cont_feats, ret['pic_labels']
                num_pseudo_instance = pseudo_instances.size(0)
                loss_contrastive = self.contrastive_loss(pseudo_instances, cont_labels)

            loss = loss_seg + loss_contrastive_weight * loss_contrastive

            # log loss weight here
            if self.is_step_to_log_for_train():
                self.cache_step_logs.append({
                    'type': "info", "value": "num_pseudo_instance: {}".format(num_pseudo_instance)
                })
                self.cache_step_tensorboard.append({
                    "type": "scalar",
                    "key": 'PseudoInstanceContrastive/NumPseudoInstance',
                    "value": num_pseudo_instance,
                    "step": self.global_step
                })

                self.cache_step_logs.append({
                    "type": "info", "value": "pic_loss_weight: {}".format(loss_contrastive_weight)
                })
                self.cache_step_tensorboard.append({
                    "type": "scalar",
                    "key": 'PseudoInstanceContrastive/LossWeight',
                    "value": loss_contrastive_weight,
                    "step": self.global_step
                })

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # wait for all the gradient to be calculated
            torch.cuda.synchronize()

            self.optimizer.step()

            if self.config.is_distributed:
                loss = reduce_tensor(loss.data, self.config.world_size)
                loss_seg = reduce_tensor(loss_seg.data, self.config.world_size)
                loss_contrastive = reduce_tensor(loss_contrastive.data, self.config.world_size)

            step_ret[self.LOSS_KEYS[0]] = loss.item()
            step_ret[self.LOSS_KEYS[1]] = loss_seg.item()
            step_ret[self.LOSS_KEYS[2]] = loss_contrastive.item()

        step_ret['semantic_scores'] = ret['semantic_scores']
        step_ret['feats'] = ret['semantic_feats']
        step_ret['pred'] = pred

        return step_ret

    def get_loss_contrastive_weight(self):
        loss_contrastive_weight = self.config.TRAINER.pseudo_instance_contrastive_loss_weight
        warmup_epoch = self.config.TRAINER.pseudo_instance_contrastive_loss_warmup_epoch
        return loss_contrastive_weight * sigmoid_rampup(self.epoch, warmup_epoch)

    def contrastive_loss(self, feats, labels):
        mode = self.config.TRAINER.pseudo_instance_contrastive_loss_mode

        if self.config.TRAINER.pseudo_instance_normalize_feat:
            feats = F.normalize(feats, dim=1)
        T = self.config.TRAINER.pseudo_instance_contrastive_temperature

        loss = None
        num_pos, num_neg, num_sample = None, None, None

        if mode == 'batch_only':
            pos_index_list = []  # (N, 1)
            neg_index_list = []  # (N, M)
            valid_idx_list = []  # (N, 1)
            # Note that N is unknown

            # has_in = False
            min_neg = float('inf')
            for label_index in range(labels.size(0)):
                pos_indexes = torch_1d_index(labels, labels[label_index])
                neg_indexes = torch_1d_neq_index(labels, labels[label_index])

                if pos_indexes.size(0) >= 2:
                    # filter self
                    _filtered_self_index = torch_1d_neq_index(pos_indexes, label_index)
                    pos_indexes = pos_indexes[_filtered_self_index]
                    # randomly select 1 as positive label
                    pos_shuffled_indexes = pos_indexes[torch.randperm(pos_indexes.size(0))]
                    pos_selected_index = pos_shuffled_indexes[0]

                    neg_len = neg_indexes.size(0)

                    # if no neg, we just ignore it
                    if neg_len < 1:
                        continue

                    if neg_len < min_neg:
                        min_neg = neg_len

                    # if self.is_step_to_log_for_train() and not has_in:
                    #     has_in = True
                    #     self.cache_step_logs.append({
                    #         'type': "info", "value": "pos_selected_index: {}".format(pos_selected_index)})

                    pos_index_list.append(pos_selected_index)
                    neg_index_list.append(neg_indexes)
                    valid_idx_list.append(label_index)

            valid_ids = torch.tensor(valid_idx_list, device=feats.device)
            pos_ids = torch.tensor(pos_index_list, device=feats.device)
            neg_ids = torch.stack([_[:min_neg] for _ in neg_index_list]).to(feats.device)

            q = feats[valid_ids, :]  # (N, d)
            k = feats[pos_ids, :]  # (N, d)
            neg = feats[neg_ids, :]  # (N, M, d)
            l_pos = torch.sum(q * k, dim=1, keepdim=True)  # (N, 1)
            l_neg = torch.bmm(q.unsqueeze(1), neg.permute([0, 2, 1])).squeeze(
                1)  # (N, 1, d) x (N, d, M) => (N, 1, M) => (N, M)

            # logits: Nx(n_neg)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= T

            contrastive_labels = torch.zeros(len(valid_idx_list), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, contrastive_labels)
            num_pos, num_neg, num_sample = 1, logits.size(1) - 1, logits.size(0)

        if mode == 'batch_only_supervised_contrastive':
            index2NegIndexes = {index: torch_1d_neq_index(labels, labels[index]) for index in range(labels.size(0))}

            # must ignore itself
            index2PosIndexes = {}
            for index in range(labels.size(0)):
                raw_indexes = torch_1d_index(labels, labels[index])
                _filtered_self_index = torch_1d_neq_index(raw_indexes, index)
                pos_indexes = raw_indexes[_filtered_self_index]
                index2PosIndexes[index] = pos_indexes

            loss = 0
            contrastive_label = torch.zeros(1, dtype=torch.long, device=feats.device)
            num_pos_record = []
            num_neg_record = []
            num_sample_record = []
            T = self.config.TRAINER.pseudo_instance_contrastive_temperature

            # compute pair-wise similarity first
            pairwise_sim = torch.matmul(feats, feats.T)  # (N, d) * (d, N) => (N, N)

            # has_in = False
            for index in range(labels.size(0)):  # TODO find more efficient impl. than for loop
                pos_indexes = index2PosIndexes[index]
                neg_indexes = index2NegIndexes[index]

                # if self.is_step_to_log_for_train() and not has_in:
                #     has_in = True
                #     self.cache_step_logs.append({
                #         'type': "info", "value": "label: {}".format(labels[index])})
                #     self.cache_step_logs.append({
                #         'type': "info", "value": "pos_labels: {}".format(labels[pos_indexes])})
                #     self.cache_step_logs.append({
                #         'type': "info", "value": "neg_labels: {}".format(labels[neg_indexes])})

                # not enough pos/neg
                if len(pos_indexes) < 1 or len(neg_indexes) < 1:
                    continue

                l_neg = pairwise_sim[index, neg_indexes].unsqueeze(0)  # (1, n_neg)
                assert l_neg.dim() == 2, 'l_neg.dim(): {}'.format(l_neg.dim())

                cur_loss = 0
                cur_weight = 1. / len(pos_indexes)
                for pos_index in pos_indexes:
                    l_pos = pairwise_sim[index, pos_index].unsqueeze(0).unsqueeze(1)  # (1, 1)
                    assert l_pos.dim() == 2, 'l_pos.dim(): {}'.format(l_pos.dim())
                    logits = torch.cat([l_pos, l_neg], dim=1)  # (1, 1+n_neg)
                    logits /= T
                    cur_loss += F.cross_entropy(logits, contrastive_label)

                loss += cur_weight * cur_loss
                num_pos_record.append(len(pos_indexes))
                num_neg_record.append(len(neg_indexes))
                num_sample_record.append(1)

            loss /= labels.size(0)
            num_pos, num_neg, num_sample = np.mean(num_pos_record), np.mean(num_neg_record), np.sum(num_sample_record)

        if mode == 'negative_bank':
            assert self.class_queue is not None
            pos_index_list = []  # (N, 1)
            valid_idx_list = []  # (N, 1)
            # Note that N is unknown

            for label_index in range(labels.size(0)):
                pos_indexes = torch_1d_index(labels, labels[label_index])

                if pos_indexes.size(0) >= 2:
                    # filter self
                    _filtered_self_index = torch_1d_neq_index(pos_indexes, label_index)
                    pos_indexes = pos_indexes[_filtered_self_index]
                    # randomly select 1 as positive label
                    pos_shuffled_indexes = pos_indexes[torch.randperm(pos_indexes.size(0))]
                    pos_selected_index = pos_shuffled_indexes[0]

                    pos_index_list.append(pos_selected_index)
                    valid_idx_list.append(label_index)

            valid_ids = torch.tensor(valid_idx_list, device=feats.device)
            pos_ids = torch.tensor(pos_index_list, device=feats.device)

            q = feats[valid_ids, :]  # (N, d)
            k = feats[pos_ids, :]  # (N, d)
            neg = torch.stack([self.class_queue.get_neg_feats(labels[_]) for _ in valid_ids], dim=0)  # (N, M, d)
            l_pos = torch.sum(q * k, dim=1, keepdim=True)  # (N, 1)
            l_neg = torch.bmm(q.unsqueeze(1), neg.permute([0, 2, 1])).squeeze(
                1)  # (N, 1, d) x (N, d, M) => (N, 1, M) => (N, M)

            # logits: Nx(n_neg)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # update class bank
            with torch.no_grad():
                self.class_queue.dequeue_and_enqueue(feats.clone().detach(), labels)

            # apply temperature
            logits /= T

            contrastive_labels = torch.zeros(len(valid_idx_list), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, contrastive_labels)
            num_pos, num_neg, num_sample = 1, logits.size(1) - 1, logits.size(0)

        if mode == 'pos_neg_bank':
            assert self.class_queue is not None

            q = feats  # (N, d)
            k = self.class_queue.get_batch_latest_pos_feats(labels)  # (N, d)
            neg = self.class_queue.get_batch_neg_feats(labels)  # (N, M, d)
            l_pos = torch.sum(q * k, dim=1, keepdim=True)  # (N, 1)
            l_neg = torch.bmm(q.unsqueeze(1), neg.permute([0, 2, 1])).squeeze(
                1)  # (N, 1, d) x (N, d, M) => (N, 1, M) => (N, M)

            # logits: Nx(n_neg)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # update class bank
            with torch.no_grad():
                self.class_queue.dequeue_and_enqueue(feats.clone().detach(), labels)

            # apply temperature
            logits /= T

            contrastive_labels = torch.zeros(q.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, contrastive_labels)
            num_pos, num_neg, num_sample = 1, logits.size(1) - 1, logits.size(0)

        assert isinstance(loss, torch.Tensor), 'got type(loss): {}, loss: {}'.format(type(loss), loss)
        assert num_pos is not None
        assert num_neg is not None
        assert num_sample is not None

        # log contrastive statistic here
        if self.is_step_to_log_for_train():
            self.cache_step_logs.append({
                'type': "info", "value": "num_contrastive_sample: {}".format(num_sample)})
            self.cache_step_logs.append({
                'type': "info", "value": "num_contrastive_negative: {}".format(num_neg)})
            self.cache_step_logs.append({
                'type': "info", "value": "num_contrastive_positive: {}".format(num_pos)})

            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'PseudoInstanceContrastive/NumSample',
                "value": num_sample,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'PseudoInstanceContrastive/NumNegative',
                "value": num_neg,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'PseudoInstanceContrastive/NumPositive',
                "value": num_pos,
                "step": self.global_step
            })

        return loss

    def get_pseudo_instance(self, coords, colors, feats, labels, predicted_scores):
        # predicted_scores: (N, num_class)
        mode = self.config.TRAINER.pseudo_instance_contrastive_pseudo_instance_mode
        predicted_scores = predicted_scores.clone().detach()
        predicted_labels = torch.argmax(predicted_scores, dim=1)  # (N, 1)
        labeled_idx = labels != self.config.DATA.ignore_label

        # expand labels or not
        if self.config.TRAINER.pseudo_instance_expand_label:
            num_expand = self.config.TRAINER.pseudo_instance_expand_label_num_per_class
            num_total_expand = 0

            # unlabeled_idx = labels == self.config.DATA.ignore_label

            # get confidence wrt the class
            predicted_scores_class = predicted_scores[torch.arange(predicted_labels.size(0)),
                                                      predicted_labels]
            predicted_scores_class_avg = torch.ones((feats.size(0),),
                                                    dtype=feats.dtype,
                                                    device=feats.device)  # max prob

            expanded_labels = labels.clone()

            for unique_class in torch.unique(predicted_labels):
                class_wise_mask = predicted_labels == unique_class
                predicted_scores_class_avg[class_wise_mask] = torch.mean(predicted_scores_class[class_wise_mask])

                # get points that are confident enough
                class_wise_mask = torch.logical_and(class_wise_mask,
                                                    predicted_scores_class > predicted_scores_class_avg)

                expandable_mask_ind = torch.nonzero(class_wise_mask.int(), as_tuple=True)[0]
                rand_ind = torch.randperm(expandable_mask_ind.size(0))[:min(num_expand,
                                                                            expandable_mask_ind.size(0))]
                expanded_labels[rand_ind] = unique_class

                num_total_expand += rand_ind.size(0)

            # recover the original label part
            expanded_labels[labeled_idx] = labels[labeled_idx]

            # update the labels with expanded_labels
            labels = expanded_labels
            labeled_idx = labels != self.config.DATA.ignore_label

            # record the actual expand
            if self.is_step_to_log_for_train():
                self.cache_step_logs.append({
                    "type": "info", "value": "pic_num_expand: {}".format(num_total_expand)
                })
                self.cache_step_tensorboard.append({
                    "type": "scalar",
                    "key": 'PseudoInstanceContrastive/NumExpand',
                    "value": num_total_expand,
                    "step": self.global_step
                })

        num_group_point = self.config.TRAINER.pseudo_instance_contrastive_num_group_point
        num_group_point_records = []
        group_threshold = self.config.TRAINER.pseudo_instance_contrastive_group_threshold
        check_label = self.config.TRAINER.pseudo_instance_contrastive_check_label

        pseudo_instances, pseudo_instance_labels = [], []
        if mode == 'euclidean_distance':
            labeled_coords = coords[labeled_idx, :]

            # coords1: (N, 1+3)
            # coords2: (M, 1+2)
            # get the point with the same batch
            batch_ids = torch.unique(coords[:, 0])

            batchId2coords = {b_id.item(): coords[coords[:, 0] == b_id, 1:] for b_id in batch_ids}
            batchId2feats = {b_id.item(): feats[coords[:, 0] == b_id, :] for b_id in batch_ids}

            for query_coord, query_label in zip(labeled_coords, labels[labeled_idx]):  # (4,)
                batch_id = query_coord[0].item()
                query_coord = query_coord[1:].unsqueeze(0)  # (1, 3)
                key_coords = batchId2coords[batch_id]  # (M, 3)
                key_feats = batchId2feats[batch_id]  # (M, d)
                dist = euclidean_distance(query_coord, key_coords)  # (1, M)

                # the closest point is itself, but we use it since avg pooling is applied
                _, closest_index = torch.topk(dist, k=num_group_point, largest=False, dim=1)  # (1, num_group_point)

                # filter the un-confident point
                if group_threshold > 0:
                    # TODO accelerate
                    # new_closest_index = [
                    #     _index for _index in closest_index[0] if predicted_scores[_index, query_label] > group_threshold
                    # ]
                    new_closest_index = closest_index[0][
                        predicted_scores[closest_index[0], query_label] > group_threshold].numpy().tolis()
                    if len(new_closest_index) == 0:
                        # append self if empty
                        # this is reasonable since self have the correct label
                        new_closest_index.append(closest_index[0][0])
                    closest_index = torch.tensor([new_closest_index], device=closest_index.device)

                if check_label:
                    # TODO accelerate
                    # new_closest_index = [
                    #     _index for _index in closest_index[0] if predicted_labels[_index] == query_label
                    # ]

                    new_closest_index = closest_index[0][
                        predicted_labels[closest_index[0]] == query_label].numpy().tolist()

                    if len(new_closest_index) == 0:
                        # append self if empty
                        # this is reasonable since self have the correct label
                        new_closest_index.append(closest_index[0][0])
                    closest_index = torch.tensor([new_closest_index], device=closest_index.device)

                num_group_point_records.append(closest_index[0].size(0))
                # TODO, attentive pooling with parameters instead of average pooling
                pseudo_instances.append(torch.mean(key_feats[closest_index[0], :], dim=0))
                pseudo_instance_labels.append(query_label)

        if mode == 'manifold':
            K = self.config.TRAINER.pseudo_instance_contrastive_manifold_K
            eta = self.config.TRAINER.pseudo_instance_contrastive_manifold_eta

            batch_ids = torch.unique(coords[:, 0])

            for batch_id in batch_ids:
                batch_id = batch_id.item()

                batch_mask = coords[:, 0] == batch_id

                scene_coords = coords[batch_mask, 1:]
                scene_colors = colors[batch_mask, :]
                scene_feats = feats[batch_mask, :]
                scene_preds = predicted_scores[batch_mask, :]
                scene_labels = labels[batch_mask]

                scene_labeled_idx = scene_labels != self.config.DATA.ignore_label

                manifold_w = get_manifold(
                    torch.cat([scene_coords[scene_labeled_idx].float(), scene_colors[scene_labeled_idx]], dim=1),
                    torch.cat([scene_coords.float(), scene_colors], dim=1), K=K, eta=eta)

                scene_labeled_range_idx = torch.nonzero(scene_labeled_idx.int(), as_tuple=True)[0]

                for _labeled_idx, _manifold_w in zip(scene_labeled_range_idx, manifold_w):
                    valid_range_idx = torch.nonzero((_manifold_w > 0).int(), as_tuple=True)[0]

                    num_group_point_records.append(valid_range_idx.size(0))

                    pseudo_instances.append(torch.mean(scene_feats[valid_range_idx, :], dim=0))
                    pseudo_instance_labels.append(scene_labels[_labeled_idx])

        assert len(pseudo_instances) > 0, 'len(pseudo_instances): {}'.format(len(pseudo_instances))

        pseudo_instances = torch.stack(pseudo_instances, dim=0)
        pseudo_instance_labels = torch.stack(pseudo_instance_labels, dim=0).long()

        assert isinstance(pseudo_instances, torch.Tensor)
        assert isinstance(pseudo_instance_labels, torch.Tensor)

        # log loss weight here
        if self.is_step_to_log_for_train():
            num_pseudo_instance = torch.sum(labeled_idx.int())
            num_avg_group_point = np.mean(num_group_point_records)
            self.cache_step_logs.append({
                'type': "info", "value": "num_pseudo_instance: {}, num_avg_group_point: {}".format(
                    num_pseudo_instance, num_avg_group_point)
            })

            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'PseudoInstanceContrastive/NumPseudoInstance',
                "value": num_pseudo_instance,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'PseudoInstanceContrastive/NumAvgGroupPoint',
                "value": num_avg_group_point,
                "step": self.global_step
            })

        return pseudo_instances, pseudo_instance_labels


def bfs_cluster_with_weak_labels(coords, labels, weak_labels, concerned_labels, ignore_label,
                                 radius=2., num_min_point=200, num_parallel=1000):
    # coords: shape (N, 3), x,y,z
    # labels: shape (N,)
    # weak_labels: shape (N,), with 255 (specified by ignore_label) to indicate missing label
    # concerned_labels: label set of interest, often indicating instance classes
    # ignore_label: fill in the weak_labels to indicate missing part

    device = labels.device

    stuff_mask = torch.zeros(coords.size(0), device=device).bool()  # init: all point are not stuff
    for concerned_label in concerned_labels:
        stuff_mask = torch.logical_or(stuff_mask, labels == concerned_label)
    # we retain all valuable label in weak_labels
    stuff_mask = torch.logical_or(stuff_mask, weak_labels != ignore_label)

    # filter out the background class directly, meanwhile, maintain a coordinate mapping
    coords = coords[stuff_mask].to(device)  # move to GPU
    labels = labels[stuff_mask]
    weak_labels = weak_labels[stuff_mask]
    partialId2fullId = torch.nonzero(stuff_mask.int(), as_tuple=True)[0].cpu()
    N = coords.size(0)  # num points

    # logging.info("===> num point to cluster: {}".format(N))

    ret_clusters = []
    ret_clusters_label = []

    visited = torch.zeros(N, device=device).bool()  # 0 indicating has not visited

    # iterative over labeled points
    valid_weak_label_mask = weak_labels != ignore_label
    not_visited_weak_label_indexes = torch.nonzero(valid_weak_label_mask.int(), as_tuple=True)[0]
    # print("===> weak_labels.size(): {}, not_visited_weak_label_indexes: {}".format(
    #     weak_labels.size(), not_visited_weak_label_indexes))

    # Note: cover the labels with corresponding weak labels
    labels[valid_weak_label_mask] = weak_labels[valid_weak_label_mask]

    for i in not_visited_weak_label_indexes.cpu().numpy().tolist():

        visited[i] = True  # also record it as well

        queue = [i]
        cluster = [i]
        while len(queue) > 0:
            # num_visited = torch.sum(visited)
            # print(
            #     "===> progress: {}% ({}/{}), len(queue): {}".format(round(float(num_visited) / N * 100, 4),
            #                                                         num_visited, N, len(queue)))
            # print("===> queue: {}".format(queue))

            num_q = min(len(queue), num_parallel)
            queue_index = torch.tensor(queue[:num_q]).long()
            queue = queue[num_q:]  # pop the num_q ele
            queue_labels = labels[queue_index]  # (num_q, )
            queue_coords = coords[queue_index, :]  # (num_q, 3)

            same_label_queue = queue_labels.unsqueeze(1) == labels.unsqueeze(0)  # (num_q, N)
            visited_queue = visited.unsqueeze(0).expand(num_q, N)  # (num_q, N)

            combined_mask = torch.logical_and(same_label_queue, ~visited_queue)  # (num_q, N)
            if torch.sum(combined_mask) < 1:  # no need to further process
                continue

            # condition at same label and not visited
            # same_label_k = labels == labels[k]  # (N,)
            # combined_mask = torch.logical_and(same_label_k, ~visited)
            # if torch.sum(combined_mask) < 1:  # no need to further process
            #     continue

            dist = euclidean_distance(queue_coords, coords)  # (num_q, N)
            # print("===> min(dist): {}".format(torch.topk(dist, k=4, largest=False)[0]))

            # FIXME only calculate distance on the valid coords
            valid_dist_k = dist < radius
            combined_mask = torch.logical_and(combined_mask, valid_dist_k)
            if torch.sum(combined_mask) < 1:  # no need to further process
                continue

            # find non zero index
            valid_indexes = torch.nonzero(combined_mask.int(), as_tuple=True)[1]

            # update related variable
            if valid_indexes.size(0) > 0:  # only update if valid
                valid_indexes = torch.unique(valid_indexes).long()
                visited[valid_indexes] = True
                valid_indexes_list = valid_indexes.cpu().numpy().tolist()
                queue.extend(valid_indexes_list)
                cluster.extend(valid_indexes_list)
                # print(
                #     "===> len(cluster): {}, num_cluster: {}, len(queue): {}, progress: {}%".format(
                #         len(cluster),
                #         len(ret_clusters),
                #         len(queue),
                #         round(float(torch.sum(visited)) / N * 100)))

        if len(cluster) > num_min_point:
            # print("===> #cluster {} found, with num point={}".format(len(ret_clusters) + 1, len(cluster)))
            ret_clusters.append(cluster)
            ret_clusters_label.append(labels[cluster[0]])

    # convert the partialId to fullId
    converted_ret_clusters = []
    for cluster in ret_clusters:
        converted_ret_clusters.append(partialId2fullId[torch.tensor(cluster).long()].numpy().tolist())

    # logging.info("===> num_cluster: {}".format(len(ret_clusters_label)))

    return converted_ret_clusters, ret_clusters_label


def bfs_cluster_v3(coords, labels, concerned_labels,
                   radius=100., num_min_point=100, num_parallel=1000):
    # coords: shape (N, 3), x,y,z
    # labels: shape (N,)

    device = labels.device

    stuff_mask = torch.zeros(coords.size(0), device=device).bool()  # init: all point are not stuff
    for concerned_label in concerned_labels:
        stuff_mask = torch.logical_or(stuff_mask, labels == concerned_label)

    # filter out the background class directly, meanwhile, maintain a coordinate mapping
    coords = coords[stuff_mask].to(device)  # move to GPU
    labels = labels[stuff_mask]
    partialId2fullId = torch.nonzero(stuff_mask.int(), as_tuple=True)[0].cpu()
    N = coords.size(0)  # num points

    # logging.info("===> num point to cluster: {}".format(N))

    ret_clusters = []
    ret_clusters_label = []

    visited = torch.zeros(N, device=device).bool()  # 0 indicating has not visited

    # speed up by non_zero
    not_visited_indexes = torch.nonzero((~visited).int(), as_tuple=True)[0]

    for i in not_visited_indexes.cpu().numpy().tolist():
        if visited[i]:
            continue

        visited[i] = True
        queue = [i]
        cluster = [i]
        while len(queue) > 0:
            # num_visited = torch.sum(visited)
            # print(
            #     "===> progress: {}% ({}/{}), len(queue): {}".format(round(float(num_visited) / N * 100, 4),
            #                                                         num_visited, N, len(queue)))
            # print("===> queue: {}".format(queue))

            num_q = min(len(queue), num_parallel)
            queue_index = torch.tensor(queue[:num_q]).long()
            queue = queue[num_q:]  # pop the num_q ele
            queue_labels = labels[queue_index]  # (num_q, )
            queue_coords = coords[queue_index, :]  # (num_q, 3)

            same_label_queue = queue_labels.unsqueeze(1) == labels.unsqueeze(0)  # (num_q, N)
            visited_queue = visited.unsqueeze(0).expand(num_q, N)  # (num_q, N)

            combined_mask = torch.logical_and(same_label_queue, ~visited_queue)  # (num_q, N)
            if torch.sum(combined_mask) < 1:  # no need to further process
                continue

            # condition at same label and not visited
            # same_label_k = labels == labels[k]  # (N,)
            # combined_mask = torch.logical_and(same_label_k, ~visited)
            # if torch.sum(combined_mask) < 1:  # no need to further process
            #     continue

            dist = euclidean_distance(queue_coords, coords)  # (num_q, N)

            # FIXME only calculate distance on the valid coords
            valid_dist_k = dist < radius
            combined_mask = torch.logical_and(combined_mask, valid_dist_k)
            if torch.sum(combined_mask) < 1:  # no need to further process
                continue

            # find non zero index
            valid_indexes = torch.nonzero(combined_mask.int(), as_tuple=True)[1]

            # update related variable
            if valid_indexes.size(0) > 0:  # only update if valid
                valid_indexes = torch.unique(valid_indexes).long()
                visited[valid_indexes] = True
                valid_indexes_list = valid_indexes.cpu().numpy().tolist()
                queue.extend(valid_indexes_list)
                cluster.extend(valid_indexes_list)
                # print(
                #     "===> len(cluster): {}, num_cluster: {}, len(queue): {}, progress: {}%".format(
                #         len(cluster),
                #         len(ret_clusters),
                #         len(queue),
                #         round(float(torch.sum(visited)) / N * 100)))

        if len(cluster) > num_min_point:
            # logging.info("===> #cluster {} found, with num point={}".format(len(ret_clusters) + 1, len(cluster)))
            ret_clusters.append(cluster)
            ret_clusters_label.append(labels[cluster[0]])

    # convert the partialId to fullId
    converted_ret_clusters = []
    for cluster in ret_clusters:
        converted_ret_clusters.append(partialId2fullId[torch.tensor(cluster).long()].numpy().tolist())

    # logging.info("===> num_cluster: {}".format(len(ret_clusters_label)))

    return converted_ret_clusters, ret_clusters_label


# TODO get weight for top K classes

def hash_classes(classes: torch.Tensor):
    assert classes.ndim == 2
    sorted_classes = torch.sort(classes, dim=1, descending=False)[0]
    # print("===> sorted_classes: {}".format(sorted_classes))

    hash_map = {}
    hash_count = 0
    hashed_classes = []

    for sorted_class in sorted_classes.tolist():
        key = tuple(sorted_class)
        if key not in hash_map:
            hash_map[key] = hash_count
            hash_count += 1
        hashed_classes.append(hash_map[key])

    hashed_classes = torch.tensor(hashed_classes, device=classes.device)

    return hash_map, hashed_classes


def bfs_cluster_multi_label(coords, topK_labels, concerned_labels,
                            radius=100., num_min_point=100, num_parallel=1000):
    # sort topK labels first
    hash_map, hashed_classes = hash_classes(topK_labels)
    reverted_hash_map = {v: k for k, v in hash_map.items()}

    # filter out the hashed_classes that include class other than concerned labels
    topK_concerned_labels = []
    concerned_labels_set = set(concerned_labels)
    for k, v in hash_map.items():
        if v not in topK_concerned_labels and set(k).issubset(concerned_labels_set):
            # logging.info("===> k: {}, v: {}, concerned: {}".format(k, v, concerned_labels_set))
            topK_concerned_labels.append(v)

    # logging.info("===> hash_map: {}".format(hash_map))
    # logging.info("===> hashed_classes: {}".format(hashed_classes))
    # logging.info("===> topK_concerned_labels: {}".format(topK_concerned_labels))

    # perform clustering
    clusters, clusters_hashed_labels = bfs_cluster_v3(coords, hashed_classes, topK_concerned_labels,
                                                      radius=radius, num_min_point=num_min_point,
                                                      num_parallel=num_parallel)

    # logging.info("===> len(clusters): {}".format(len(clusters)))
    # logging.info("===> len(clusters_hashed_labels): {}".format(len(clusters_hashed_labels)))
    # logging.info("===> clusters_hashed_labels: {}".format(clusters_hashed_labels))
    # covert clusters_hashed_labels
    clusters_labels = [list(reverted_hash_map[_.item()]) for _ in clusters_hashed_labels]
    clusters_labels = torch.tensor(clusters_labels, device=topK_labels.device)

    return clusters, clusters_labels


def partition_coords_and_logits(coords, logits, entropy_mask):
    return coords[entropy_mask], logits[entropy_mask]


@torch.no_grad()
def all_to_batch_bfs_cluster(all_coords, all_labels, concerned_labels,
                             radius=100., num_min_point=100, num_parallel=1000,
                             multi_label=False, all_weak_labels=None, ignore_label=None):
    if all_weak_labels is not None and multi_label is True:
        raise ValueError("weak labels mode is not implemented for multi-label clustering")
    if all_weak_labels is not None and ignore_label is None:
        raise ValueError("ignore_label is required for weak labels mode")

    cluster_func = bfs_cluster_v3 if not multi_label else bfs_cluster_multi_label

    batch_ids = torch.unique(all_coords[:, 0])

    batch_clusters = []  # (num_batch, num_cluster, num_point)
    batch_clusters_labels = []  # (num_batch, num_cluster)

    for batch_id in batch_ids:
        batch_id = batch_id.item()

        batch_mask = all_coords[:, 0] == batch_id

        coords = all_coords[batch_mask, 1:]
        labels = all_labels[batch_mask]

        # clusters: not a tensor, a list of lists, (num_cluster, num_point)
        # clusters_labels: not a tensor, a list, (num_cluster,)

        if all_weak_labels is None:
            clusters, clusters_labels = cluster_func(coords, labels, concerned_labels,
                                                     radius=radius,
                                                     num_min_point=num_min_point,
                                                     num_parallel=num_parallel)
        else:
            weak_labels = all_weak_labels[batch_mask]
            clusters, clusters_labels = bfs_cluster_with_weak_labels(
                coords, labels, weak_labels, concerned_labels,
                radius=radius,
                num_min_point=num_min_point,
                num_parallel=num_parallel,
                ignore_label=ignore_label)
        batch_clusters.append(clusters)
        batch_clusters_labels.append(clusters_labels)

    return batch_clusters, batch_clusters_labels


def all_to_batch_scores(all_coords, all_scores):
    batch_ids = torch.unique(all_coords[:, 0])

    batch_scores = []  # (num_batch, num_point, num_class)

    for batch_id in batch_ids:
        batch_id = batch_id.item()

        batch_mask = all_coords[:, 0] == batch_id
        batch_scores.append(all_scores[batch_mask, :])
    return batch_scores


def get_num_cluster_and_its_point(batch_clusters):
    num_cluster = sum([len(_) for _ in batch_clusters])
    num_cluster_point = sum([len(_) for clusters in batch_clusters for _ in clusters])

    return num_cluster, num_cluster_point


@TRAINER_REGISTRY.register()  # shortname: Getaway?
class GeometricAwareGlobalLocalTrainer(FullySupervisedTrainer):
    LOSS_KEYS = FullySupervisedTrainer.LOSS_KEYS

    def __init__(self, config):
        super(GeometricAwareGlobalLocalTrainer, self).__init__(config)

        self.instance_categories = self.get_instance_categories()
        self.background_categories = self.get_background_categories()

        self.entropy_threshold = self.config.TRAINER.geo_aware_entropy_threshold
        self.radius = self.config.TRAINER.geo_aware_bfs_cluster_radius
        self.num_min_point = self.config.TRAINER.geo_aware_bfs_cluster_num_min_point
        self.num_parallel = self.config.TRAINER.geo_aware_bfs_cluster_num_parallel

        self.warmup_epoch = self.config.TRAINER.geo_aware_loss_warmup_epoch
        self.loss_seg_conf_apply = self.config.TRAINER.geo_aware_loss_seg_conf_apply
        self.loss_seg_conf_weight = self.config.TRAINER.geo_aware_loss_seg_conf_weight
        self.loss_consistency_conf_apply = self.config.TRAINER.geo_aware_loss_consistency_conf_apply
        self.loss_consistency_conf_weight = self.config.TRAINER.geo_aware_loss_consistency_conf_weight
        self.K = self.config.TRAINER.geo_aware_loss_seg_diff_K
        self.loss_seg_diff_apply = self.config.TRAINER.geo_aware_loss_seg_diff_apply
        self.loss_seg_diff_weight = self.config.TRAINER.geo_aware_loss_seg_diff_weight
        self.loss_consistency_diff_apply = self.config.TRAINER.geo_aware_loss_consistency_diff_apply
        self.loss_consistency_diff_key = self.config.TRAINER.geo_aware_loss_consistency_diff_key
        self.loss_consistency_diff_weight = self.config.TRAINER.geo_aware_loss_consistency_diff_weight
        self.loss_prototype_apply = self.config.TRAINER.geo_aware_loss_prototype_apply
        self.loss_prototype_weight = self.config.TRAINER.geo_aware_loss_prototype_weight
        self.loss_contrastive_apply = self.config.TRAINER.geo_aware_loss_contrastive_apply
        self.loss_contrastive_weight = self.config.TRAINER.geo_aware_loss_contrastive_weight
        self.loss_contrastive_key = self.config.TRAINER.geo_aware_loss_contrastive_key
        self.loss_contrastive_t = self.config.TRAINER.geo_aware_loss_contrastive_t

        self.vis_cluster = self.config.TRAINER.geo_aware_vis_cluster

        self.cluster_cache = {}  # some variables may be used multiple times by different loss functions

        # update the loss key by flag
        for apply_flag, apply_str in zip([
            self.loss_seg_conf_apply,
            self.loss_consistency_conf_apply,
            self.loss_seg_diff_apply,
            self.loss_consistency_diff_apply,
            self.loss_prototype_apply,
            self.loss_contrastive_apply,
        ], [
            'LossSegConf',
            'LossConsisConf',
            'LossSegDiff',
            'LossConsisDiff',
            'LossPrototype',
            'LossCont',
        ]):
            if apply_flag:
                self.LOSS_KEYS.append(apply_str)

    def step(self, batch):
        coords = batch['coords']
        feats = batch['feats']
        targets = None
        if 'labels' in batch:
            targets = self.to_cuda_if_use(batch['labels'])

        if self.config.AUGMENTATION.normalize_color:
            assert self.config.AUGMENTATION.normalize_coord is False
            feats[:, :3] = feats[:, :3] / 255. - 0.5
        if self.config.AUGMENTATION.normalize_coord:  # for semantic kitti
            assert self.config.AUGMENTATION.normalize_color is False
            feats[:, 0] = (feats[:, 0] - (-0.3527016)) / 14.5789787
            feats[:, 1] = (feats[:, 1] - 0.7280641) / 9.84908962
            feats[:, 2] = (feats[:, 2] - (-0.96655365)) / 0.90581832

        sparse_input = self.to_cuda_if_use(ME.SparseTensor(feats, coords))

        ret = self.model(sparse_input, targets, global_step=self.global_step, hack_mlp=self.epoch < self.warmup_epoch)

        step_ret = {}

        if self.model.training:
            # add indexes for cluster visualization
            self.cluster_cache['indexes'] = batch['indexes']
            # add rgb feats for vis
            rgb_feats = (feats[:, :3] + 0.5) * 255.
            self.cluster_cache['all_rgb_feats'] = rgb_feats
            # add weak labels as well
            self.cluster_cache['all_weak_labels'] = targets

            loss_seg = ret['loss']
            loss_rets = {'LossSeg': loss_seg}
            loss_weights = {'LossSeg': 1.0}

            any_geo_loss = self.loss_seg_conf_apply \
                           or self.loss_consistency_conf_apply \
                           or self.loss_seg_diff_apply \
                           or self.loss_consistency_diff_key \
                           or self.loss_contrastive_apply
            is_the_right_epoch = self.epoch >= self.warmup_epoch

            suffix = ''
            if is_the_right_epoch and any_geo_loss:
                history_apply = self.config.MODEL.look_back_model_apply
                mlp_head_apply = self.config.MODEL.mlp_head_model_apply
                if history_apply and mlp_head_apply:
                    raise ValueError("history model and mlp head model are not compatible")

                if history_apply:
                    # clustering with history feats
                    geo_feats = ret['semantic_feats_history']
                    geo_scores = ret['semantic_scores_history']
                else:
                    geo_feats = ret['semantic_feats']
                    geo_scores = ret['semantic_scores']

                    if self.loss_contrastive_apply:
                        geo_feats = ret[self.loss_contrastive_key]

                # _e = get_entropy(torch.softmax(ret['semantic_scores'], dim=1))
                # logging.info("current entropy size: {}, entropy mean: {}".format(_e.size(), torch.mean(_e)))

                self.prepare_geo_data(coords, geo_scores, geo_feats)

                if history_apply:
                    # back prob the latest feats
                    self.update_geo_data(ret['semantic_scores'], ret['semantic_feats'])
                    suffix = '_update'

            if self.loss_seg_conf_apply and is_the_right_epoch:
                # no ignore label
                loss_rets['LossSegConf'] = self.cluster_cross_entropy_loss(
                    self.cluster_cache['batch_clusters_conf'],
                    self.cluster_cache['batch_scores_conf{}'.format(suffix)],
                    self.cluster_cache['batch_clusters_labels_conf'])
                loss_weights['LossSegConf'] = self.get_loss_weight(self.loss_seg_conf_weight)

            if self.loss_consistency_conf_apply and is_the_right_epoch:
                loss_rets['LossConsisConf'] = self.cluster_consistency_loss(
                    self.cluster_cache['batch_clusters_conf'],
                    self.cluster_cache['batch_scores_conf{}'.format(suffix)])
                loss_weights['LossConsisConf'] = self.get_loss_weight(self.loss_consistency_conf_weight)

            if self.loss_seg_diff_apply and is_the_right_epoch:
                assert 'prototypes' in ret, 'prototypes are needed in geo trainer, ' \
                                            'please set config.MODEL.prototype_model_apply=True'

                prototypes = ret['prototypes']
                loss_rets['LossSegDiff'] = self.cluster_cross_entropy_loss_with_prototypes(
                    self.cluster_cache['batch_clusters_diff'],
                    self.cluster_cache['batch_scores_diff{}'.format(suffix)],
                    self.cluster_cache['batch_feats_diff{}'.format(suffix)],
                    prototypes,
                    self.cluster_cache['batch_clusters_labels_diff'])
                loss_weights['LossSegDiff'] = self.get_loss_weight(self.loss_seg_diff_weight)

            if self.loss_consistency_diff_apply and is_the_right_epoch:
                assert self.loss_consistency_diff_key in self.cluster_cache
                k = self.loss_consistency_diff_key
                loss_rets['LossConsisDiff'] = self.cluster_consistency_loss(
                    self.cluster_cache['batch_clusters_diff'],
                    self.cluster_cache['{}{}'.format(k, suffix)])
                loss_weights['LossConsisDiff'] = self.get_loss_weight(self.loss_consistency_diff_weight)

            if self.loss_prototype_apply:
                assert 'prototypes' in ret, 'prototypes are needed in geo trainer, ' \
                                            'please set config.MODEL.prototype_model_apply=True'

                prototypes = ret['prototypes']
                prototype_logits = torch.matmul(ret['semantic_feats'], prototypes.T)  # (num_point, num_class)
                loss_rets['LossPrototype'] = F.cross_entropy(prototype_logits, targets,
                                                             ignore_index=self.config.DATA.ignore_label)
                loss_weights['LossPrototype'] = self.get_loss_weight(self.loss_prototype_weight)

            if self.loss_contrastive_apply and is_the_right_epoch:
                loss_rets['LossCont'] = self.cluster_contrastive_loss(
                    self.cluster_cache['batch_clusters_wl'],
                    self.cluster_cache['batch_feats'],
                    self.cluster_cache['batch_clusters_labels_wl'])
                loss_weights['LossCont'] = self.get_loss_weight(self.loss_contrastive_weight)

            # aggregate loss here
            total_loss = 0
            for loss_key in self.LOSS_KEYS:
                if loss_key not in loss_rets:
                    continue
                # logging.info(
                #     "===> loss_key: {}, loss_rets[loss_key].size(): {}".format(loss_key, loss_rets[loss_key].size()))

                total_loss += (loss_weights[loss_key] * loss_rets[loss_key])
            # final loss is here
            loss_rets['Loss'] = total_loss
            assert not isinstance(total_loss, int)

            # log loss and its weight here
            if self.is_step_to_log_for_train():
                for loss_key in self.LOSS_KEYS:
                    if loss_key not in loss_rets or loss_key not in loss_weights:
                        continue
                    self.cache_step_logs.append({
                        "type": "info",
                        "value": "geo_{}_weight: {}".format(get_lower_case_name(loss_key), loss_weights[loss_key])
                    })
                    self.cache_step_tensorboard.append({
                        "type": "scalar",
                        "key": 'GeometricAwareGlobalLocal/{}Weight'.format(loss_key),
                        "value": loss_weights[loss_key],
                        "step": self.global_step
                    })

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()

            # wait for all the gradient to be calculated
            torch.cuda.synchronize()

            self.optimizer.step()

            if self.config.is_distributed:
                # aggregate loss for loss logging
                for loss_key in self.LOSS_KEYS:
                    if loss_key not in loss_rets:
                        continue
                    loss_data = reduce_tensor(loss_rets[loss_key].data, self.config.world_size)
                    step_ret[loss_key] = loss_data.item()

        step_ret['semantic_scores'] = ret['semantic_scores']
        step_ret['feats'] = ret['semantic_feats']
        step_ret['pred'] = torch.argmax(step_ret['semantic_scores'], dim=1)

        # empty cache
        self.clear_geo_data()

        return step_ret

    def get_params_dict(self):
        pic_params = super().get_params_dict()

        trainer_config = self.config.TRAINER
        for k in trainer_config:
            if k.startswith('geo_aware_'):
                pic_params[k] = trainer_config[k]

        return pic_params

    def get_instance_categories(self):
        if 'Scannet' in self.config.DATA.dataset:
            return self.config.DATA.scannet_instance_categories
        if 'Stanford' in self.config.DATA.dataset:
            return self.config.DATA.stanford_instance_categories
        raise ValueError("Unsupported dataset found: {}".format(self.config.DATA.dataset))

    def get_background_categories(self):
        if 'Scannet' in self.config.DATA.dataset:
            return self.config.DATA.scannet_background_categories
        if 'Stanford' in self.config.DATA.dataset:
            return self.config.DATA.stanford_background_categories
        raise ValueError("Unsupported dataset found: {}".format(self.config.DATA.dataset))

    def cluster_cross_entropy_loss(self,
                                   batch_clusters,
                                   batch_scores,
                                   batch_clusters_labels,
                                   multi_label=False):

        _l1, _l2, _l3 = len(batch_clusters), len(batch_scores), len(batch_clusters_labels)
        assert _l1 == _l2 == _l3, "{}, {}, {}, does not equal".format(_l1, _l2, _l3)

        collected_cluster_scores, collected_cluster_labels = [], []
        # iterate over batch dimension
        for clusters, scores, clusters_labels in zip(batch_clusters, batch_scores, batch_clusters_labels):
            for cluster, cluster_label in zip(clusters, clusters_labels):
                cluster_scores = scores[torch.tensor(cluster).long(), :]  # (len(cluster), 13)
                if not multi_label:
                    cluster_labels = torch.tensor([cluster_label] * len(cluster), dtype=torch.long, device=self.device)
                else:
                    # cluster_label (3) => (1, 3) => (len(cluster), 3)
                    # logging.info("===> cluster_label: {}, cluster_label.size(): {}".format(
                    #     cluster_label, cluster_label.size()
                    # ))
                    cluster_labels = cluster_label.unsqueeze(0).repeat(len(cluster), 1)
                collected_cluster_scores.append(cluster_scores)
                collected_cluster_labels.append(cluster_labels)

        if len(collected_cluster_scores) == 0:
            return torch.zeros(1, device=self.device, dtype=torch.float).squeeze()

        # concat along point dimension
        collected_cluster_scores = torch.cat(collected_cluster_scores, dim=0)  # (num_cluster*len(cluster), 13)
        collected_cluster_labels = torch.cat(collected_cluster_labels, dim=0)  # (num_cluster*len(cluster), 1 or 3)
        # logging.info("===> collected_cluster_scores.size(): {}".format(collected_cluster_scores.size()))
        # logging.info("===> collected_cluster_labels.size(): {}".format(collected_cluster_labels.size()))
        if not multi_label:
            # y_p, y_t
            # 1, 1
            # 2, 2
            # 3, 3
            # mse(y_p, y_t) = (y_p - y_t) ** 2 = 0
            # loss * delta(loss) / delta(\theta) = 0
            return F.cross_entropy(collected_cluster_scores, collected_cluster_labels)  # no ignore label here
        else:
            loss = 0
            K = collected_cluster_labels.size(1)
            for k in range(K):
                loss += F.cross_entropy(collected_cluster_scores, collected_cluster_labels[:, k])
            return loss / float(K)

    def cluster_cross_entropy_loss_with_prototypes(self,
                                                   batch_clusters,
                                                   batch_scores,
                                                   batch_feats,
                                                   prototypes,  # (num_class, dim)
                                                   batch_clusters_labels):
        # Note: only applicable for multi label
        # TODO: assign label by prototypes instead of the cluster label

        _l1, _l2, _l3, _l4 = len(batch_clusters), len(batch_scores), len(batch_clusters_labels), len(batch_feats)
        assert _l1 == _l2 == _l3 == _l4, "{}, {}, {}, {} does not equal".format(_l1, _l2, _l3, _l4)

        collected_cluster_scores, collected_cluster_labels = [], []
        # iterate over batch dimension
        for clusters, scores, feats, clusters_labels in zip(
                batch_clusters,
                batch_scores,
                batch_feats,
                batch_clusters_labels
        ):
            for cluster, cluster_label in zip(clusters, clusters_labels):  # cluster_label: (3,)
                cluster_indexes = torch.tensor(cluster).long()
                cluster_scores = scores[cluster_indexes, :]  # (len(cluster), 13)
                cluster_feats = feats[cluster_indexes, :]  # (len(cluster), dim)

                # assign label here
                with torch.no_grad():
                    cluster_feats_mean = torch.mean(cluster_feats, dim=0, keepdim=True)  # (1, dim)
                    cluster_prototypes = prototypes[cluster_label]  # (3, dim)
                    sim = torch.matmul(cluster_prototypes, cluster_feats_mean.T).squeeze(1)  # (3, 1) => (3,)
                    assigned_label_index = torch.argmin(sim)  # (1,)
                    assigned_cluster_label = cluster_label[assigned_label_index]  # (1,)
                cluster_labels = assigned_cluster_label.repeat(len(cluster))  # (len(cluster))

                if self.is_step_to_log_for_train():
                    logging.info("===> assigned_label_index: {}, assigned_cluster_label: {} ".format(
                        assigned_label_index, assigned_cluster_label))
                    # logging.info(
                    #     "===> assigned_cluster_label: {}, assigned_cluster_label.size(): {}".format(
                    #         assigned_cluster_label,
                    #         assigned_cluster_label.size()))

                    # logging.info(
                    #     "===> cluster_labels: {}, cluster_labels.size(): {}".format(
                    #         cluster_labels, cluster_labels.size()))

                collected_cluster_scores.append(cluster_scores)
                collected_cluster_labels.append(cluster_labels)

        if len(collected_cluster_scores) == 0:
            return torch.zeros(1, device=self.device, dtype=torch.float).squeeze()

        # concat along point dimension
        collected_cluster_scores = torch.cat(collected_cluster_scores, dim=0)  # (num_cluster*len(cluster), 13)
        collected_cluster_labels = torch.cat(collected_cluster_labels, dim=0)  # (num_cluster*len(cluster), 1)
        # logging.info("===> collected_cluster_scores.size(): {}".format(collected_cluster_scores.size()))
        # logging.info("===> collected_cluster_labels.size(): {}".format(collected_cluster_labels.size()))

        loss = F.cross_entropy(collected_cluster_scores, collected_cluster_labels)
        return loss

    def cluster_consistency_loss(self, batch_clusters, batch_scores):
        _l1, _l2 = len(batch_clusters), len(batch_scores)
        assert _l1 == _l2, "{}, {}, does not equal".format(_l1, _l2)

        collected_cluster_scores, collected_cluster_centers = [], []
        for clusters, scores in zip(batch_clusters, batch_scores):
            for cluster in clusters:
                cluster_scores = scores[torch.tensor(cluster).long(), :]
                with torch.no_grad():  # no grad to gt
                    cluster_center = torch.mean(cluster_scores, dim=0, keepdim=True).repeat(len(cluster), 1)
                collected_cluster_scores.append(cluster_scores)
                collected_cluster_centers.append(cluster_center)

        if len(collected_cluster_scores) == 0:
            return torch.zeros(1, device=self.device, dtype=torch.float).squeeze()

        # concat along point dimension
        collected_cluster_scores = torch.cat(collected_cluster_scores, dim=0)  # (num_point, num_class)
        collected_cluster_centers = torch.cat(collected_cluster_centers, dim=0)  # (num_point, num_class)
        # logging.info("===> collected_cluster_scores.size(): {}".format(collected_cluster_scores.size()))
        # logging.info("===> collected_cluster_centers.size(): {}".format(collected_cluster_centers.size()))
        return F.mse_loss(collected_cluster_scores, collected_cluster_centers)

    def is_valid_pseudo_instance_label(self, labels):
        is_valid = True

        # filter out the invalid labels
        filtered_labels = labels.clone()

        label_unique, label_count = torch.unique(filtered_labels[filtered_labels != self.config.DATA.ignore_label],
                                                 return_counts=True)

        # filter out not instance category
        label_unique = torch_intersect(label_unique.cpu(), torch.tensor(self.instance_categories))

        # only consider instance category

        # three case
        # (0) if no label available
        if label_unique.size(0) == 0:
            is_valid = False
        # (1) if all the same, no_neg
        if label_unique.size(0) == 1:
            is_valid = False
        # (2) if all unique, no_pos
        if label_unique.size(0) == label_count.sum():
            is_valid = False

        if not is_valid:
            logging.warning("Invalid pseudo instance label found, label_unique={}".format(label_unique))

        return is_valid

    def cluster_contrastive_loss(self,
                                 batch_clusters,
                                 batch_feats,
                                 batch_clusters_labels):

        cont_feats, labels = [], []  # (num_ins, dim), (num_ins)
        for clusters, feats, clusters_labels in zip(batch_clusters,
                                                    batch_feats,
                                                    batch_clusters_labels):
            for cluster, cluster_label in zip(clusters, clusters_labels):
                cluster_feats = torch.mean(feats[cluster], dim=0)
                cont_feats.append(cluster_feats)
                labels.append(cluster_label)

        if len(cont_feats) == 0:
            return 1e-7 * torch.nn.functional.smooth_l1_loss(
                batch_feats[0], torch.zeros_like(batch_feats[0]).fill_(torch.mean(batch_feats[0]))
            )

        cont_feats = torch.stack(cont_feats, dim=0)
        labels = torch.stack(labels, dim=0)
        logging.info("===> labels: {}".format(labels))

        if not self.is_valid_pseudo_instance_label(labels):
            return 1e-7 * torch.nn.functional.smooth_l1_loss(
                batch_feats[0], torch.zeros_like(batch_feats[0]).fill_(torch.mean(batch_feats[0]))
            )

        cont_feats = F.normalize(cont_feats, dim=1)

        # compute pair-wise similarity first
        pairwise_sim = torch.matmul(cont_feats, cont_feats.T)  # (N, d) * (d, N) => (N, N)
        index2NegIndexes = {index: torch_1d_neq_index(labels, labels[index]) for index in range(labels.size(0))}

        # must ignore itself
        index2PosIndexes = {}
        for index in range(labels.size(0)):
            raw_indexes = torch_1d_index(labels, labels[index])
            _filtered_self_index = torch_1d_neq_index(raw_indexes, index)
            pos_indexes = raw_indexes[_filtered_self_index]
            index2PosIndexes[index] = pos_indexes

        loss = torch.zeros(1, device=self.device).squeeze()
        contrastive_label = torch.zeros(1, dtype=torch.long, device=cont_feats.device)
        num_pos_record = []
        num_neg_record = []
        num_sample_record = []
        T = self.config.TRAINER.pseudo_instance_contrastive_temperature

        # has_in = False
        for index in range(labels.size(0)):  # TODO find more efficient impl. than for loop
            pos_indexes = index2PosIndexes[index]
            neg_indexes = index2NegIndexes[index]

            # if self.is_step_to_log_for_train() and not has_in:
            #     has_in = True
            #     self.cache_step_logs.append({
            #         'type': "info", "value": "label: {}".format(labels[index])})
            #     self.cache_step_logs.append({
            #         'type': "info", "value": "pos_labels: {}".format(labels[pos_indexes])})
            #     self.cache_step_logs.append({
            #         'type': "info", "value": "neg_labels: {}".format(labels[neg_indexes])})

            # not enough pos/neg
            if len(pos_indexes) < 1 or len(neg_indexes) < 1:
                continue

            l_neg = pairwise_sim[index, neg_indexes].unsqueeze(0)  # (1, n_neg)
            assert l_neg.dim() == 2, 'l_neg.dim(): {}'.format(l_neg.dim())

            cur_loss = 0
            cur_weight = 1. / len(pos_indexes)
            for pos_index in pos_indexes:
                l_pos = pairwise_sim[index, pos_index].unsqueeze(0).unsqueeze(1)  # (1, 1)
                assert l_pos.dim() == 2, 'l_pos.dim(): {}'.format(l_pos.dim())
                logits = torch.cat([l_pos, l_neg], dim=1)  # (1, 1+n_neg)
                logits /= T
                cur_loss += F.cross_entropy(logits, contrastive_label)

            loss += cur_weight * cur_loss
            num_pos_record.append(len(pos_indexes))
            num_neg_record.append(len(neg_indexes))
            num_sample_record.append(1)

        loss /= labels.size(0)
        num_pos, num_neg, num_sample = np.mean(num_pos_record), np.mean(num_neg_record), np.sum(num_sample_record)

        if self.is_step_to_log_for_train():
            self.cache_step_logs.append({
                'type': "info",
                "value": "geo_contrast_num_pos: {}, geo_contrast_num_neg: {}, geo_contrast_num_sample: {}".format(
                    num_pos, num_neg, num_sample
                )
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumPositive',
                "value": num_pos,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumNegative',
                "value": num_neg,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumSample',
                "value": num_sample,
                "step": self.global_step
            })

        return loss

    def get_loss_weight(self, weight):
        return weight * sigmoid_rampup(self.epoch, self.warmup_epoch)

    def geo_loss(self, all_coords, all_scores, all_feats, prototypes, all_rgb_feats=None):
        # scores, before softmax
        # feats, features before classification layer
        # rgb_feats, rgb, for visualization and debug

        # Note: all_xxx means all xxx in one batch, to distinguish with batch_xxx

        # DEBUG, variables to save to visualize
        # (1) all_coords, all_scores, all_feats
        # (2) all_entropy_mask
        # (3) batch_scores_conf, batch_scores_diff
        # (4) batch_clusters_conf, batch_clusters_labels_conf
        # (5) batch_clusters_diff, batch_clusters_labels_diff

        ret = {}

        # start grouping when network are trained after warmup_epoch
        if self.epoch < self.warmup_epoch:
            return ret

        with torch.no_grad():
            # (1) prepare data
            # 1.1 get pseudo label
            all_pseudo_labels = torch.argmax(all_scores, dim=1)
            all_topK_pseudo_labels = torch.topk(all_scores, k=self.K)[1]

            # 1.2 get entropy and entropy map
            all_entropy = get_entropy(torch.softmax(all_scores, dim=1))
            all_entropy_mask = get_confident_entropy_mask(all_entropy, self.entropy_threshold)

        # (2) partition data into two parts
        # 2.1 confident part
        all_coords_conf = all_coords[all_entropy_mask]
        all_scores_conf = all_scores[all_entropy_mask]
        all_pseudo_labels_conf = all_pseudo_labels[all_entropy_mask]
        all_feats_conf = all_feats[all_entropy_mask]

        # 2.2 diffident part
        all_coords_diff = all_coords[~all_entropy_mask]
        all_scores_diff = all_scores[~all_entropy_mask]
        all_topK_pseudo_labels_diff = all_topK_pseudo_labels[~all_entropy_mask]
        all_feats_diff = all_feats[~all_entropy_mask]

        # (3) cluster & loss computation
        # 3.1 cluster confident points
        batch_clusters_conf, batch_clusters_labels_conf = all_to_batch_bfs_cluster(all_coords_conf,
                                                                                   all_pseudo_labels_conf,
                                                                                   self.instance_categories,
                                                                                   self.radius,
                                                                                   self.num_min_point,
                                                                                   self.num_parallel)
        # logging.info("===> batch_clusters_labels_conf: {}".format(batch_clusters_labels_conf))
        batch_scores_conf = all_to_batch_scores(all_coords_conf, all_scores_conf)
        batch_feats_conf = all_to_batch_scores(all_coords_conf, all_feats_conf)

        # 3.2 cluster diffident points
        batch_clusters_diff, batch_clusters_labels_diff = all_to_batch_bfs_cluster(all_coords_diff,
                                                                                   all_topK_pseudo_labels_diff,
                                                                                   self.instance_categories,
                                                                                   self.radius,
                                                                                   self.num_min_point,
                                                                                   self.num_parallel,
                                                                                   multi_label=True)
        # logging.info("===> batch_clusters_labels_diff: {}".format(batch_clusters_labels_diff))
        batch_scores_diff = all_to_batch_scores(all_coords_diff, all_scores_diff)
        batch_feats_diff = all_to_batch_scores(all_coords_diff, all_feats_diff)

        # FIXME visualization only
        if self.is_step_to_log_for_train() and self.vis_cluster:
            filedir = os.path.join(self.config.exp_path, 'vis_cluster')
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            save_dict = {
                'all_coords': all_coords,
                'all_scores': all_scores,
                'all_feats': all_rgb_feats,
                'all_entropy_mask': all_entropy_mask,  # no need to save it, we can compute it
                'batch_scores_conf': batch_scores_conf,
                'batch_scores_diff': batch_scores_diff,
                'batch_clusters_conf': batch_clusters_conf,
                'batch_clusters_labels_conf': batch_clusters_labels_conf,
                'batch_clusters_diff': batch_clusters_diff,
                'batch_clusters_labels_diff': batch_clusters_labels_diff,

            }
            save_path = os.path.join(filedir, 'batch_cluster_step_{}.pth'.format(self.global_step))
            logging.info('Saving cluster visualization to {}...'.format(save_path))
            torch.save(save_dict, save_path)

        # 3.3 cross entropy loss for confident clusters
        if self.loss_seg_conf_apply:
            loss_seg_conf = self.cluster_cross_entropy_loss(batch_clusters_conf,
                                                            batch_scores_conf,
                                                            batch_clusters_labels_conf)  # no ignore label
            ret['LossSegConf'] = loss_seg_conf

        # 3.4 consistency loss among points inside cluster
        if self.loss_consistency_conf_apply:
            loss_consistency_conf = self.cluster_consistency_loss(batch_clusters_conf,
                                                                  batch_scores_conf)
            ret['LossConsisConf'] = loss_consistency_conf

        # 3.5 cross entry loss for diffident clusters
        if self.loss_seg_diff_apply:
            # logging.info("===> batch_clusters_labels_diff: \n{}".format(batch_clusters_labels_diff))
            loss_seg_diff = self.cluster_cross_entropy_loss_with_prototypes(batch_clusters_diff,
                                                                            batch_scores_diff,
                                                                            batch_feats_diff,
                                                                            prototypes,
                                                                            batch_clusters_labels_diff)
            ret['LossSegDiff'] = loss_seg_diff

        # 3.6 consistency loss among points inside cluster
        if self.loss_consistency_diff_apply:
            loss_consistency_diff = self.cluster_consistency_loss(batch_clusters_diff,
                                                                  batch_scores_diff)
            ret['LossConsisDiff'] = loss_consistency_diff

        if self.is_step_to_log_for_train():
            num_confident_point = torch.sum(all_entropy_mask.int()).item()
            confident_percentage = 100. * num_confident_point / all_entropy_mask.size(0)
            avg_entropy = torch.mean(all_entropy).item()
            num_cluster_conf, num_cluster_point_conf = get_num_cluster_and_its_point(batch_clusters_conf)
            num_cluster_diff, num_cluster_point_diff = get_num_cluster_and_its_point(batch_clusters_diff)

            self.cache_step_logs.append({
                'type': "info",
                "value": "num_confident_point: {}, "
                         "confident_percentage: {}, "
                         "avg_entropy: {}, "
                         "num_cluster_conf: {}, "
                         "num_cluster_point_conf: {}, "
                         "num_cluster_diff: {}, "
                         "num_cluster_point_diff: {}".format(
                    num_confident_point, confident_percentage, avg_entropy,
                    num_cluster_conf, num_cluster_point_conf,
                    num_cluster_diff, num_cluster_point_diff,
                )
            })

            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumConfPoints',
                "value": num_confident_point,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/ConfPercent',
                "value": confident_percentage,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/AvgEntropy',
                "value": avg_entropy,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterConf',
                "value": num_cluster_conf,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterPointConf',
                "value": num_cluster_point_conf,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterDiff',
                "value": num_cluster_diff,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterPointDiff',
                "value": num_cluster_point_diff,
                "step": self.global_step
            })

        return ret

    @torch.no_grad()
    def prepare_shared_clustering_cache(self, all_scores):
        # Note: cache aim to calculate only once

        #  get pseudo label
        if 'all_pseudo_labels' not in self.cluster_cache:
            self.cluster_cache['all_pseudo_labels'] = torch.argmax(all_scores, dim=1)
        # get top k pseudo label
        if 'all_topK_pseudo_labels' not in self.cluster_cache:
            all_topK_pseudo_labels = torch.topk(all_scores, k=self.K)[1]
            self.cluster_cache['all_topK_pseudo_labels'] = all_topK_pseudo_labels

        # get entropy and entropy map
        if 'all_entropy' not in self.cluster_cache:
            self.cluster_cache['all_entropy'] = get_entropy(torch.softmax(all_scores, dim=1))
        all_entropy = self.cluster_cache['all_entropy']
        if 'all_entropy_mask' not in self.cluster_cache:
            self.cluster_cache['all_entropy_mask'] = get_confident_entropy_mask(all_entropy, self.entropy_threshold)

    def prepare_conf_clustering_cache(self, all_coords, all_scores, all_feats):

        # prepare shared clustering cache first
        self.prepare_shared_clustering_cache(all_scores)

        all_entropy_mask = self.cluster_cache['all_entropy_mask']
        all_pseudo_labels = self.cluster_cache['all_pseudo_labels']

        # confident part
        if 'all_coords_conf' not in self.cluster_cache:
            self.cluster_cache['all_coords_conf'] = all_coords[all_entropy_mask]
        if 'all_scores_conf' not in self.cluster_cache:
            self.cluster_cache['all_scores_conf'] = all_scores[all_entropy_mask]
        if 'all_pseudo_labels_conf' not in self.cluster_cache:
            self.cluster_cache['all_pseudo_labels_conf'] = all_pseudo_labels[all_entropy_mask]
        if 'all_feats_conf' not in self.cluster_cache:
            self.cluster_cache['all_feats_conf'] = all_feats[all_entropy_mask]

        # cluster confident points
        if 'batch_clusters_conf' not in self.cluster_cache or 'batch_clusters_labels_conf' not in self.cluster_cache:
            all_coords_conf = self.cluster_cache['all_coords_conf']
            all_pseudo_labels_conf = self.cluster_cache['all_pseudo_labels_conf']
            batch_clusters_conf, batch_clusters_labels_conf = all_to_batch_bfs_cluster(all_coords_conf,
                                                                                       all_pseudo_labels_conf,
                                                                                       self.instance_categories,
                                                                                       self.radius,
                                                                                       self.num_min_point,
                                                                                       self.num_parallel)
            self.cluster_cache['batch_clusters_conf'] = batch_clusters_conf
            self.cluster_cache['batch_clusters_labels_conf'] = batch_clusters_labels_conf
            # logging.info("===> batch_clusters_labels_conf: {}".format(batch_clusters_labels_conf))

        if 'batch_scores_conf' not in self.cluster_cache or 'batch_feats_conf' not in self.cluster_cache:
            all_coords_conf = self.cluster_cache['all_coords_conf']
            all_scores_conf = self.cluster_cache['all_scores_conf']
            all_feats_conf = self.cluster_cache['all_feats_conf']
            batch_scores_conf = all_to_batch_scores(all_coords_conf, all_scores_conf)
            batch_feats_conf = all_to_batch_scores(all_coords_conf, all_feats_conf)
            self.cluster_cache['batch_scores_conf'] = batch_scores_conf
            self.cluster_cache['batch_feats_conf'] = batch_feats_conf

    def prepare_diff_clustering_cache(self, all_coords, all_scores, all_feats):
        # prepare shared clustering cache first
        self.prepare_shared_clustering_cache(all_scores)

        all_entropy_mask = self.cluster_cache['all_entropy_mask']
        all_topK_pseudo_labels = self.cluster_cache['all_topK_pseudo_labels']

        # diffident part
        if 'all_coords_diff' not in self.cluster_cache:
            self.cluster_cache['all_coords_diff'] = all_coords[~all_entropy_mask]
        if 'all_scores_diff' not in self.cluster_cache:
            self.cluster_cache['all_scores_diff'] = all_scores[~all_entropy_mask]
        if 'all_topK_pseudo_labels_diff' not in self.cluster_cache:
            self.cluster_cache['all_topK_pseudo_labels_diff'] = all_topK_pseudo_labels[~all_entropy_mask]
        if 'all_feats_diff' not in self.cluster_cache:
            self.cluster_cache['all_feats_diff'] = all_feats[~all_entropy_mask]

        # cluster diffident points
        if 'batch_clusters_diff' not in self.cluster_cache or 'batch_clusters_labels_diff' not in self.cluster_cache:
            all_coords_diff = self.cluster_cache['all_coords_diff']
            all_topK_pseudo_labels_diff = self.cluster_cache['all_topK_pseudo_labels_diff']
            batch_clusters_diff, batch_clusters_labels_diff = all_to_batch_bfs_cluster(all_coords_diff,
                                                                                       all_topK_pseudo_labels_diff,
                                                                                       self.instance_categories,
                                                                                       self.radius,
                                                                                       self.num_min_point,
                                                                                       self.num_parallel,
                                                                                       multi_label=True)
            self.cluster_cache['batch_clusters_diff'] = batch_clusters_diff
            self.cluster_cache['batch_clusters_labels_diff'] = batch_clusters_labels_diff
            # logging.info("===> batch_clusters_labels_diff: {}".format(batch_clusters_labels_diff))
        if 'batch_scores_diff' not in self.cluster_cache or 'batch_feats_diff' not in self.cluster_cache:
            all_coords_diff = self.cluster_cache['all_coords_diff']
            all_scores_diff = self.cluster_cache['all_scores_diff']
            all_feats_diff = self.cluster_cache['all_feats_diff']
            batch_scores_diff = all_to_batch_scores(all_coords_diff, all_scores_diff)
            batch_feats_diff = all_to_batch_scores(all_coords_diff, all_feats_diff)
            self.cluster_cache['batch_scores_diff'] = batch_scores_diff
            self.cluster_cache['batch_feats_diff'] = batch_feats_diff

    def prepare_all_clustering_by_weak_labels_cache(self, all_coords, all_scores, all_feats):
        # cluster all points by weak labels, wl denotes weak labels
        if 'batch_clusters_wl' not in self.cluster_cache or 'batch_clusters_labels_wl' not in self.cluster_cache:
            all_pseudo_labels = self.cluster_cache['all_pseudo_labels']
            all_weak_labels = self.cluster_cache['all_weak_labels']

            batch_clusters_wl, batch_clusters_labels_wl = all_to_batch_bfs_cluster(
                all_coords, all_pseudo_labels, self.instance_categories,
                self.radius, self.num_min_point, self.num_parallel,
                all_weak_labels=all_weak_labels,
                ignore_label=self.config.DATA.ignore_label)

            self.cluster_cache['batch_clusters_wl'] = batch_clusters_wl
            self.cluster_cache['batch_clusters_labels_wl'] = batch_clusters_labels_wl

        if 'batch_feats' not in self.cluster_cache or 'batch_scores' not in self.cluster_cache:
            self.cluster_cache['batch_scores'] = all_to_batch_scores(all_coords, all_scores)
            self.cluster_cache['batch_feats'] = all_to_batch_scores(all_coords, all_feats)

    @torch.no_grad()
    def log_geo_data(self):

        if 'has_logged' in self.cluster_cache and self.cluster_cache['has_logged'] is True:
            return  # already logged

        if 'all_entropy_mask' in self.cluster_cache:
            all_entropy_mask = self.cluster_cache['all_entropy_mask']
            num_confident_point = torch.sum(all_entropy_mask.int()).item()
            confident_percentage = 100. * num_confident_point / all_entropy_mask.size(0)

            self.cache_step_logs.append({
                'type': "info",
                "value": "num_confident_point: {}, confident_percentage: {}".format(
                    num_confident_point, confident_percentage
                )
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumConfPoints',
                "value": num_confident_point,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/ConfPercent',
                "value": confident_percentage,
                "step": self.global_step
            })

        if 'all_entropy' in self.cluster_cache:
            all_entropy = self.cluster_cache['all_entropy']
            avg_entropy = torch.mean(all_entropy).item()
            self.cache_step_logs.append({
                'type': "info",
                "value": "avg_entropy: {}".format(avg_entropy)
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/AvgEntropy',
                "value": avg_entropy,
                "step": self.global_step
            })

        if 'batch_clusters_conf' in self.cluster_cache:
            batch_clusters_conf = self.cluster_cache['batch_clusters_conf']
            num_cluster_conf, num_cluster_point_conf = get_num_cluster_and_its_point(batch_clusters_conf)
            self.cache_step_logs.append({
                'type': "info",
                "value": "num_cluster_conf: {}, num_cluster_point_conf: {}".format(
                    num_cluster_conf, num_cluster_point_conf,
                )
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterConf',
                "value": num_cluster_conf,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterPointConf',
                "value": num_cluster_point_conf,
                "step": self.global_step
            })

        if 'batch_clusters_diff' in self.cluster_cache:
            batch_clusters_diff = self.cluster_cache['batch_clusters_diff']
            num_cluster_diff, num_cluster_point_diff = get_num_cluster_and_its_point(batch_clusters_diff)
            self.cache_step_logs.append({
                'type': "info",
                "value": "num_cluster_diff: {}, num_cluster_point_diff: {}".format(
                    num_cluster_diff, num_cluster_point_diff,
                )
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterDiff',
                "value": num_cluster_diff,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterPointDiff',
                "value": num_cluster_point_diff,
                "step": self.global_step
            })

        if 'batch_clusters_wl' in self.cluster_cache:
            batch_clusters_wl = self.cluster_cache['batch_clusters_wl']
            num_cluster_wl, num_cluster_point_wl = get_num_cluster_and_its_point(batch_clusters_wl)
            self.cache_step_logs.append({
                'type': "info",
                "value": "num_cluster_wl: {}, num_cluster_point_wl: {}".format(
                    num_cluster_wl, num_cluster_point_wl,
                )
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterWL',
                "value": num_cluster_wl,
                "step": self.global_step
            })
            self.cache_step_tensorboard.append({
                "type": "scalar",
                "key": 'GeometricAwareGlobalLocal/NumClusterPointWL',
                "value": num_cluster_point_wl,
                "step": self.global_step
            })

        # FIXME visualization only
        if self.is_step_to_log_for_train() and self.vis_cluster:
            filedir = os.path.join(self.config.exp_path, 'vis_cluster')
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            save_dict = {
                'indexes': self.cluster_cache.get('indexes', None),
                'all_coords': self.cluster_cache.get('all_coords', None),
                'all_scores': self.cluster_cache.get('all_scores', None),
                'all_feats': self.cluster_cache.get('all_feats', None),
                'all_rgb_feats': self.cluster_cache.get('all_rgb_feats', None),
                'all_weak_labels': self.cluster_cache.get('all_weak_labels', None),
                # # no need to save it, we can compute it
                'all_entropy_mask': self.cluster_cache.get('all_entropy_mask', None),
                'batch_scores_conf': self.cluster_cache.get('batch_scores_conf', None),
                'batch_scores_diff': self.cluster_cache.get('batch_scores_diff', None),
                'batch_clusters_conf': self.cluster_cache.get('batch_clusters_conf', None),
                'batch_clusters_labels_conf': self.cluster_cache.get('batch_clusters_labels_conf', None),
                'batch_clusters_diff': self.cluster_cache.get('batch_clusters_diff', None),
                'batch_clusters_labels_diff': self.cluster_cache.get('batch_clusters_labels_diff', None),
                'batch_clusters_wl': self.cluster_cache.get('batch_clusters_wl', None),
                'batch_clusters_labels_wl': self.cluster_cache.get('batch_clusters_labels_wl', None),
            }
            index_str = ''
            if self.cluster_cache.get('indexes', None) is not None:
                index_str = '_' + '.'.join([str(_) for _ in self.cluster_cache['indexes'].tolist()])
            save_path = os.path.join(filedir, 'batch_cluster_step_{}{}.pth'.format(self.global_step, index_str))
            logging.info('Saving cluster visualization to {}...'.format(save_path))
            torch.save(save_dict, save_path)

        self.cluster_cache['has_logged'] = True

    def prepare_geo_data(self, all_coords, all_scores, all_feats):

        if 'all_coords' not in self.cluster_cache:
            self.cluster_cache['all_coords'] = all_coords
        if 'all_scores' not in self.cluster_cache:
            self.cluster_cache['all_scores'] = all_scores
        if 'all_feats' not in self.cluster_cache:
            self.cluster_cache['all_feats'] = all_feats

        # prepare commonly shared part
        self.prepare_shared_clustering_cache(all_scores)

        # prepare conf part if loss required
        if self.loss_seg_conf_apply or self.loss_consistency_conf_apply:
            self.prepare_conf_clustering_cache(all_coords, all_scores, all_feats)

        # prepare diff part if loss required
        if self.loss_seg_diff_apply or self.loss_consistency_diff_apply:
            self.prepare_diff_clustering_cache(all_coords, all_scores, all_feats)

        # prepare all part if loss required
        if self.loss_contrastive_apply:
            self.prepare_all_clustering_by_weak_labels_cache(all_coords, all_scores, all_feats)

        # log geo data
        if self.is_step_to_log_for_train():
            self.log_geo_data()

    def update_conf_cache(self, all_scores_update, all_feats_update):
        if 'batch_scores_conf_update' not in self.cluster_cache or 'batch_feats_conf_update' not in self.cluster_cache:
            all_coords_conf = self.cluster_cache['all_coords_conf']
            all_entropy_mask = self.cluster_cache['all_entropy_mask']
            all_scores_conf_update = all_scores_update[all_entropy_mask]
            all_feats_conf_update = all_feats_update[all_entropy_mask]
            batch_scores_conf_update = all_to_batch_scores(all_coords_conf, all_scores_conf_update)
            batch_feats_conf_update = all_to_batch_scores(all_coords_conf, all_feats_conf_update)
            self.cluster_cache['batch_scores_conf_update'] = batch_scores_conf_update
            self.cluster_cache['batch_feats_conf_update'] = batch_feats_conf_update

    def update_diff_cache(self, all_scores_update, all_feats_update):
        if 'batch_scores_diff_update' not in self.cluster_cache or 'batch_feats_diff_update' not in self.cluster_cache:
            all_coords_diff = self.cluster_cache['all_coords_diff']
            all_entropy_mask = self.cluster_cache['all_entropy_mask']
            all_scores_diff_update = all_scores_update[~all_entropy_mask]
            all_feats_diff_update = all_feats_update[~all_entropy_mask]
            batch_scores_diff_update = all_to_batch_scores(all_coords_diff, all_scores_diff_update)
            batch_feats_diff_update = all_to_batch_scores(all_coords_diff, all_feats_diff_update)
            self.cluster_cache['batch_scores_diff_update'] = batch_scores_diff_update
            self.cluster_cache['batch_feats_diff_update'] = batch_feats_diff_update

    def update_geo_data(self, all_scores_update, all_feats_update):  # coords are the same, no need to update
        # update conf part if loss required
        if self.loss_seg_conf_apply or self.loss_consistency_conf_apply:
            self.update_conf_cache(all_scores_update, all_feats_update)

        # update diff part if loss required
        if self.loss_seg_diff_apply or self.loss_consistency_diff_apply:
            self.update_diff_cache(all_scores_update, all_feats_update)

    def clear_geo_data(self):
        self.cluster_cache.clear()

    def handle_distributed_parallel(self, model):
        config = self.config

        if self.use_cuda:
            logging.info('Using cuda: {} to accelerate model training'.format(config.rank))
            model = model.to(config.rank)

        # TODO handle norm sync bn for me.bn

        if config.is_distributed:
            logging.info('Converting model to DDP model')
            """
            RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
            This error indicates that your module has parameters that were not used in producing loss.
            """
            # So we do a little hack here to resolve the above issue
            # which only move the train model to ddp
            # and leave the history model intact
            if not self.config.MODEL.look_back_model_apply:
                model = DDP(
                    module=model, device_ids=[self.device],
                    output_device=self.device,
                    broadcast_buffers=False
                )
            else:
                _model = model.model
                _model = DDP(
                    module=_model, device_ids=[self.device],
                    output_device=self.device,
                    broadcast_buffers=False
                )
                model.model = _model
            logging.info('Done Converting model to DDP model')
            # DDP(model, [config.rank])
        else:
            model = model

        return model
