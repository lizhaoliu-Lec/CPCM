import logging
import os
from pathlib import Path

import numpy as np
import torch
from fvcore.common.registry import Registry

import dataset.fully_supervised.transforms as t
from dataset.fully_supervised.base import VoxelizationDataset, DatasetPhase, str2datasetphase_type, \
    DataLoaderForFactory, ForVisualizeDataLoaderForFactory
from lib.utils import read_txt
from ..build import DATASET_REGISTRY

SEMANTIC_KITTI = Registry("SemanticKitti")
SEMANTIC_KITTI.__doc__ = """
Registry for module that is compatible with semantic_kitti dataloader.

The registered object will be called with `obj(cfg)`.
"""


@SEMANTIC_KITTI.register()
class SemanticKittiVoxelizationDataset(VoxelizationDataset):
    # added
    NUM_LABELS = 19
    NUM_IN_CHANNEL = 3
    CLASS_LABELS = (
        'car', 'bicycle', 'motorcycle', 'truck', 'othervehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
        'parking', 'sidewalk', 'otherground', 'building', 'fence', 'vegetation',
        'trunk', 'terrain', 'pole', 'trafficsign')
    VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    CLASS_LABELS_INSTANCE = ['car', 'bicycle', 'motorcycle', 'truck', 'othervehicle', 'person', 'bicyclist',
                             'motorcyclist',
                             'trunk', 'pole', 'trafficsign']
    VALID_CLASS_IDS_INSTANCE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 15, 17, 18])
    IGNORE_LABELS_INSTANCE = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE))

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = None  # TODO use elastic augmentation or not

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    IS_FULL_POINTCLOUD_EVAL = True

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'semantic_kitti_train.txt',
        DatasetPhase.Val: 'semantic_kitti_val.txt',
        DatasetPhase.TrainVal: 'semantic_kitti_train.txt',
        DatasetPhase.Test: 'semantic_kitti_val.txt',
    }

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 phase=DatasetPhase.Train):
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        # Use cropped rooms for train/val
        data_root = config.DATA.semantic_kitti_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND

        data_paths = read_txt(os.path.join(data_root, 'splits', self.DATA_PATH_FILE[phase]))
        if phase == DatasetPhase.Train and config.DATA.train_file:
            data_paths = read_txt(config.DATA.train_file)

        # data efficiency by sampling points
        self.sampled_inds = {}
        sampled_inds_file = config.DATA.semantic_kitti_sampled_inds
        if sampled_inds_file and phase == DatasetPhase.Train:
            logging.info("Loading sampled_ids from {}".format(sampled_inds_file))
            self.sampled_inds = torch.load(sampled_inds_file)

        data_paths = [data_path + '.pth' for data_path in data_paths]
        logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.DATA.ignore_label,
            return_transformation=config.DATA.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config,
            phase=phase)

    def get_output_id(self, index):
        return '_'.join(Path(self.data_paths[index]).stem.split('_')[:2])

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
             pointcloud[:, 6:]))
        return pointcloud

    def load_data(self, index):
        filepath = self.data_root / self.data_paths[index]
        pointcloud = torch.load(filepath)
        coords = pointcloud[:, :3].astype(np.float32)
        feats = pointcloud[:, 3:6].astype(np.float32)
        if self.phase == DatasetPhase.Test:
            labels = torch.zeros(coords.shape[0]) + self.ignore_mask
            # instance_labels = torch.zeros(coords.shape[0]) + self.ignore_mask
        else:
            labels = pointcloud[:, 6].astype(np.int32)
            # instance_labels = pointcloud[3].astype(np.int32)
            if self.sampled_inds:
                scene_name = self.get_output_id(index)
                mask = np.ones_like(labels).astype(np.bool)
                sampled_inds = self.sampled_inds[scene_name]
                mask[sampled_inds] = False
                labels[mask] = self.ignore_mask  # set it to ignore label
        # hack the intensity to rgb here
        # intensity in range (0, 1)
        # print("===> coords.size(): {}".format(coords.shape))
        return coords, feats, labels, np.ones(coords.shape[0])

    def extra_repr(self):
        return "num_classes={}, voxel_size={}".format(len(self.VALID_CLASS_IDS), self.VOXEL_SIZE)


@SEMANTIC_KITTI.register()
class SemanticKittiVoxelization2cmDataset(SemanticKittiVoxelizationDataset):
    VOXEL_SIZE = 0.02


@DATASET_REGISTRY.register()
class SemanticKittiDataLoader(DataLoaderForFactory):
    DATASET_CLASS = SEMANTIC_KITTI
