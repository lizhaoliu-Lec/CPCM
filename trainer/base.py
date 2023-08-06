import glob
import logging
import os
import pprint
import random
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Union

import numpy as np
import torch

from tensorboardX import SummaryWriter
from tensorboardX.writer import numpy_compatible

from meta_data.constant import PROJECT_NAME
from meta_data.scan_net import LABEL_to_COLOR_and_NAME
from trainer.lr_scheduler import build_lr_scheduler
from util.eval import EVALUATOR_REGISTRY
from util.visualize import load_ply, draw_point_cloud_plolty
from util import eval

from dataset import build_dataset
from model import build_model
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = ['BaseTrainer']


def build_optimizer(cfg, model):
    logger = logging.getLogger(PROJECT_NAME)
    logger.info('Using optimizer {}'.format(cfg.OPTIMIZER.name))
    params = dict(vars(cfg.OPTIMIZER))
    params.pop('name')
    logger.info('Optimizer params: {}'.format(params))
    optimizer_factory = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
    }
    if cfg.OPTIMIZER.name not in optimizer_factory:
        raise ValueError('Unsupported optimizer `{}`'.format(cfg.OPTIMIZER.name))
    return optimizer_factory[cfg.OPTIMIZER.name](params=filter(lambda p: p.requires_grad, model.parameters()), **params)


class BuildEvaluator:
    def __init__(self, cfg):
        self.logger = logging.getLogger(PROJECT_NAME)
        self.logger.info('Using evaluator {}'.format(cfg.EVALUATOR.metrics))
        self.config = cfg
        self.factory = {
            name: EVALUATOR_REGISTRY.get(name)(cfg) for name in cfg.EVALUATOR.metrics
        }

    def clear_all(self):
        for evaluator in self.factory.values():
            evaluator.clear()

    def update_all(self, pred: torch.Tensor, gt: torch.Tensor, batch_offsets: torch.Tensor):
        for evaluator in self.factory.values():
            evaluator.update(pred, gt, batch_offsets)

    def get_all_value(self) -> dict:
        result = {}
        for evaluator in self.factory.values():
            result.update(evaluator.get_value())
        return result


def count_param(_model):
    return sum([x.nelement() for x in _model.parameters()])


class DistributedSummaryWriter(SummaryWriter):

    def __init__(self, cfg):
        self.config = cfg
        self._initialized = False
        if cfg.rank == 0:
            super().__init__(cfg.exp_path)
            self._initialized = True

    def add_scalar(self, *args, **kwargs):
        if self._initialized:
            super().add_scalar(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self._initialized:
            super().add_histogram(*args, **kwargs)


class BaseTrainer:
    def __init__(self, config):
        self.config = config

        logger = logging.getLogger(PROJECT_NAME)
        logger.info("Using trainer as follow: {}".format(self))

        self.writer = None

        # ===========> init env and backup files
        self.init()

        # ===========> init train, val, test dataloader
        logger.info('Initializing dataset...')
        self.dataset = build_dataset(config)

        self.handle_after_dataset()

        # ===========> init model, resume parameters
        self.start_epoch = self.epoch = 0
        self.global_step = 0
        self.is_best = False
        self.metric_dict = {}

        # build model and print info
        logger.info('Initializing model...')
        model = build_model(config)
        logger.debug('Model initialized as follow: \n{}'.format(model))
        num_params = count_param(model)
        logger.info('The #Params of model is: {} ({:.2f}M)'.format(num_params, num_params / 1e6))

        # move to cuda if available
        self.use_cuda = torch.cuda.is_available()

        self.model = self.handle_distributed_parallel(model)

        self.handle_after_model()

        # resume model if specified
        model_config = config.MODEL
        if hasattr(model_config, 'resume') and model_config.resume is True:
            assert hasattr(model_config, 'resume_path')
            self.resume_checkpoint()

        # init optimizer
        logger.info('Initializing optimizer...')
        self.optimizer = self.build_optimizer()

        # init lr scheduler
        logger.info('Initializing lr scheduler...')
        self.lr_scheduler = self.build_lr_scheduler()

        # init evaluator
        logger.info('Initializing evaluator...')
        self.evaluator = BuildEvaluator(config)

    def init(self):
        logger = logging.getLogger(PROJECT_NAME)
        config = self.config

        # backup file
        if self.config.rank == 0:
            logger.info('Backing up files...')
            backup_dir = os.path.join(config.exp_path, 'backup_files')
            os.makedirs(backup_dir, exist_ok=True)
            # backup trainer, model, dataset, and config
            os.system('cp -r {} {}'.format('trainer/', backup_dir))
            os.system('cp -r {} {}'.format('model/', backup_dir))
            os.system('cp -r {} {}'.format('dataset/', backup_dir))
            os.system('cp {} {}'.format(config.config_path, backup_dir))
            logger.info('Backing up files done!')

        # init writer
        logger.info('Initializing summary writer...')
        self.writer = DistributedSummaryWriter(config)

        # set up torch random seed
        seed = config.GENERAL.manual_seed

        if self.config.is_distributed:
            seed += self.config.rank

        logger.info('Fixing rand seed = {}...'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True  # TODO how it affect speed?
        # torch.backends.cudnn.benchmark = False

    def train(self):
        logger = logging.getLogger(PROJECT_NAME)
        for epoch in range(self.start_epoch, self.config.TRAINER.epochs):
            logger.info('Start training for epoch {}...'.format(epoch + 1))
            self.epoch = epoch

            torch.cuda.empty_cache()  # empty cache from the last epoch
            # train one epoch
            self.train_one_epoch()

            logger.info('Start validation for epoch {}...'.format(epoch + 1))
            self.val()

            self.lr_scheduler.step(epoch=epoch, step_type='epoch')

            # TODO test should be called by selecting the best model validated by validation set
            # logger.info('Start test for epoch {}...'.format(epoch))
            # self.test()

    def train_one_epoch(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def resume_checkpoint(self):
        logger = logging.getLogger(PROJECT_NAME)

        # check if file exists
        resume_path = self.config.MODEL.resume_path
        if not os.path.exists(resume_path) or not os.path.isfile(resume_path):
            raise ValueError('File `{}` does not exist'.format(resume_path))
        logger.info('Resuming model parameters from path: {}'.format(resume_path))
        loaded_dict = torch.load(resume_path, map_location={"cuda:0": "cuda:{}".format(self.config.rank)})

        if 'model' not in loaded_dict:
            self.model.load_state_dict(loaded_dict)
        else:
            #  The key of loaded model do not match the key of current model
            model_dict = OrderedDict()
            for k, v in loaded_dict["model"].items():
                name = k[13:]
                name = '.'.join(["module", name])
                model_dict[name] = v
            self.model.load_state_dict(model_dict)
            self.start_epoch = self.epoch = loaded_dict['epoch'] + 1  # begin from next epoch
            self.global_step = loaded_dict['global_step'] if 'global_step' in loaded_dict else 0
            self.is_best = loaded_dict['is_best']
            self.metric_dict = loaded_dict['metric_dict']

    def save_checkpoint(self, metric_dict, epoch, is_best=False):
        if self.config.rank != 0:
            return

        logger = logging.getLogger(PROJECT_NAME)

        save_dict = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'is_best': is_best,
            'metric_dict': metric_dict,
        }
        save_path = os.path.join(self.config.exp_path, 'model_latest.pth')
        logger.info('Saving latest model to {}...'.format(save_path))
        torch.save(save_dict, save_path)
        logger.info('Saving latest model done')

        if is_best:
            # save_path = os.path.join(self.config.exp_path, 'model_best.pth')
            best_save_path = os.path.join(self.config.exp_path, 'model_best.pth')
            logger.info('Saving best model to {}... <===> best'.format(best_save_path))
            # torch.save(save_dict, save_path)
            # use copy to improve performance
            shutil.copy(save_path, best_save_path)
            logger.info('Saving best model done')

    def to_cuda_if_use(self, _d):
        return _d.cuda() if self.use_cuda else _d

    def build_optimizer(self):
        return build_optimizer(self.config, self.model)

    def build_lr_scheduler(self):
        return build_lr_scheduler(self.config, self.optimizer)

    def handle_distributed_parallel(self, model):
        config = self.config

        if self.use_cuda:
            logging.info('Using cuda: {} to accelerate model training'.format(config.rank))
            model = model.to(config.rank)

        logging.info('Using norm_fn: {}'.format(config.MODEL.norm_fn.name))
        if config.MODEL.norm_fn.name == 'BN' and config.MODEL.norm_fn.use_sync_bn and config.is_distributed:
            logging.info('Converting batch norm layer to sync batch norm layer')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if config.is_distributed:
            logging.info('Converting model to DDP model')
            model = DDP(model, [config.rank])
        else:
            model = model

        return model

    def handle_after_dataset(self):
        pass

    def handle_after_model(self):
        pass

    def __repr__(self):
        params = self.get_params_dict()

        return '{}(\n{}\n)'.format(self.__class__.__name__, pprint.pformat(params))

    def get_params_dict(self):
        return {
            'epochs': self.config.TRAINER.epochs,
            'log_every': self.config.TRAINER.log_every
        }
