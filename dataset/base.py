import logging

from meta_data.constant import PROJECT_NAME


class BaseDataset:
    def __init__(self, cfg):
        self.batch_size = cfg.DATA.batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size
        if hasattr(cfg.DATA, 'val_batch_size'):
            self.val_batch_size = cfg.DATA.val_batch_size
        if hasattr(cfg.DATA, 'test_batch_size'):
            self.test_batch_size = cfg.DATA.test_batch_size

        self.num_workers = cfg.DATA.num_workers
        self.num_val_workers = self.num_workers
        self.num_test_workers = self.num_workers
        if hasattr(cfg.DATA, 'num_val_workers'):
            self.num_val_workers = cfg.DATA.num_val_workers
        if hasattr(cfg.DATA, 'num_test_workers'):
            self.num_test_workers = cfg.DATA.num_test_workers

        # get logger
        self.logger = logging.getLogger(PROJECT_NAME)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
