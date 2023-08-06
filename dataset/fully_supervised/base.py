import logging
import time
from abc import ABC
from pathlib import Path
from collections import defaultdict

import numpy as np
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

from plyfile import PlyData
import dataset.fully_supervised.transforms as t
from dataset.base import BaseDataset
from dataset.fully_supervised.sampler import InfSampler, DistributedInfSampler
from dataset.fully_supervised.voxelizer import Voxelizer
from lib.distributed import get_world_size

from util.me_compatible import IS_OLD_ME
from torch.utils.data._utils.collate import default_collate


class DatasetPhase(Enum):
    Train = 0
    Val = 1
    Val2 = 2
    TrainVal = 3
    Test = 4
    Debug = 5


def datasetphase_2str(arg):
    if arg == DatasetPhase.Train:
        return 'train'
    elif arg == DatasetPhase.Val:
        return 'val'
    elif arg == DatasetPhase.Val2:
        return 'val2'
    elif arg == DatasetPhase.TrainVal:
        return 'trainval'
    elif arg == DatasetPhase.Test:
        return 'test'
    elif arg == DatasetPhase.Debug:
        return 'debug'
    else:
        raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
    if arg.upper() == 'TRAIN':
        return DatasetPhase.Train
    elif arg.upper() == 'VAL':
        return DatasetPhase.Val
    elif arg.upper() == 'VAL2':
        return DatasetPhase.Val2
    elif arg.upper() == 'TRAINVAL':
        return DatasetPhase.TrainVal
    elif arg.upper() == 'TEST':
        return DatasetPhase.Test
    elif arg.upper() == 'DEBUG':
        return DatasetPhase.Debug
    else:
        raise ValueError('phase must be one of train/val/test')


def cache(func):
    def wrapper(self, *args, **kwargs):
        # Assume that args[0] is an index
        index = args[0]
        if self.cache:
            if index not in self.cache_dict[func.__name__]:
                results = func(self, *args, **kwargs)
                self.cache_dict[func.__name__][index] = results
            return self.cache_dict[func.__name__][index]
        else:
            return func(self, *args, **kwargs)

    return wrapper


class DictDataset(Dataset, ABC):
    IS_FULL_POINTCLOUD_EVAL = False

    def __init__(self,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 data_root='/'):
        """
        data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
        """
        Dataset.__init__(self)

        # Allows easier path concatenation
        if not isinstance(data_root, Path):
            data_root = Path(data_root)

        self.data_root = data_root
        self.data_paths = sorted(data_paths)

        self.prevoxel_transform = prevoxel_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

        # dictionary of input
        self.data_loader_dict = {
            'input': (self.load_input, self.input_transform),
            'target': (self.load_target, self.target_transform)
        }

        # For large dataset, do not cache
        self.cache = cache
        self.cache_dict = defaultdict(dict)
        self.loading_key_order = ['input', 'target']

    def load_input(self, index):
        raise NotImplementedError

    def load_target(self, index):
        raise NotImplementedError

    def get_classnames(self):
        pass

    def reorder_result(self, result):
        return result

    def __getitem__(self, index):
        out_array = []
        for k in self.loading_key_order:
            loader, transformer = self.data_loader_dict[k]
            v = loader(index)
            if transformer:
                v = transformer(v)
            out_array.append(v)
        return out_array

    def __len__(self):
        return len(self.data_paths)


class VoxelizationDatasetBase(DictDataset, ABC):
    IS_TEMPORAL = False
    CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
    ROTATION_AXIS = None
    NUM_IN_CHANNEL = None
    NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
    IGNORE_LABELS = None  # labels that are not evaluated

    def __init__(self,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 data_root='/',
                 ignore_mask=255,
                 return_transformation=False,
                 **kwargs):
        """
        ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
        """
        DictDataset.__init__(
            self,
            data_paths,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            cache=cache,
            data_root=data_root)

        self.ignore_mask = ignore_mask
        self.return_transformation = return_transformation

    def __getitem__(self, index):
        raise NotImplementedError

    def load_ply(self, index):
        filepath = self.data_root / self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        return coords, feats, labels, None

    def load_data(self, index):
        raise NotImplementedError

    def __len__(self):
        num_data = len(self.data_paths)
        return num_data


class VoxelizationDataset(VoxelizationDatasetBase):
    """This dataset loads RGB point clouds and their labels as a list of points
    and voxelizes the pointcloud with sufficient data augmentation.
    """
    _repr_indent = 4

    # Voxelization arguments
    VOXEL_SIZE = 0.05  # 5cm

    # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
    # augmentation has to be done before voxelization
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
    ELASTIC_DISTORT_PARAMS = None

    # MISC.
    PREVOXELIZATION_VOXEL_SIZE = None

    # Augment coords to feats
    AUGMENT_COORDS_TO_FEATS = False

    def __init__(self,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 data_root='/',
                 ignore_label=255,
                 return_transformation=False,
                 augment_data=False,
                 config=None,
                 phase=DatasetPhase.Train,
                 **kwargs):
        is_train_phase = phase == DatasetPhase.Train or phase == DatasetPhase.TrainVal

        self.augment_data = augment_data
        self.config = config
        self.two_stream = config.DATA.two_stream
        # only retain label during training, do not alter the validation settings
        self.retain_label = self.config.DATA.retain_label and is_train_phase

        VoxelizationDatasetBase.__init__(
            self,
            data_paths,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            cache=cache,
            data_root=data_root,
            ignore_mask=ignore_label,
            return_transformation=return_transformation)

        # Prevoxel transformations
        self.voxelizer = Voxelizer(
            voxel_size=self.VOXEL_SIZE,
            clip_bound=self.CLIP_BOUND,
            use_augmentation=augment_data,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
            ignore_label=ignore_label,
            sparse_label=self.config.DATA.sparse_label,  # TODO sparse label is also set for validation set
            retain_label=self.retain_label,
        )

        # map labels not evaluated to ignore_label
        self.label_map_array = self.gen_label_map_v2()  # v2, an array, speed up

        self.NUM_LABELS -= len(self.IGNORE_LABELS)
        self.phase = phase

    def gen_label_map_v2(self):
        label_map_array = np.array([self.ignore_mask] * self.NUM_LABELS, dtype=np.int)
        # ignore some label
        preserved_label_indexes = np.array([_ for _ in range(self.NUM_LABELS) if _ not in self.IGNORE_LABELS],
                                           dtype=np.int)
        preserved_label_idxes = np.array([_ for _ in range(self.NUM_LABELS - len(self.IGNORE_LABELS))])
        label_map_array[preserved_label_indexes] = preserved_label_idxes
        return label_map_array

    def map_label_v2(self, labels):
        new_labels = np.zeros_like(labels) + self.ignore_mask
        valid_mask = labels != self.ignore_mask
        new_labels[valid_mask] = self.label_map_array[labels[valid_mask]]
        return new_labels

    def _augment_coords_to_feats(self, coords, feats, labels=None, idx2hasPoint=None):
        norm_coords = coords - coords.mean(0)
        # color must come first.
        if isinstance(coords, np.ndarray):
            feats = np.concatenate((feats, norm_coords), 1)
        else:
            feats = torch.cat((feats, norm_coords), 1)
        return coords, feats, labels, idx2hasPoint

    def convert_mat2cfl(self, mat):
        # Generally, xyz,rgb,label
        return mat[:, :3], mat[:, 3:-1], mat[:, -1]

    def get_instance_info(self, xyz, instance_ids):
        '''
        :param xyz: (n, 3)
        :param instance_ids: (n), int, (1~nInst, -1)
        :return: instance_num, dict
        '''
        centers = np.ones((xyz.shape[0], 3),
                          dtype=np.float32) * -1  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz, occ, num_instances)
        occupancy = {}  # (nInst), int
        bbox = {}
        unique_ids = np.unique(instance_ids)
        for id_ in unique_ids:
            if id_ == -1:
                continue

            mask = (instance_ids == id_)
            xyz_ = xyz[mask]
            bbox_min = xyz_.min(0)
            bbox_max = xyz_.max(0)
            center = xyz_.mean(0)

            centers[mask] = center
            occupancy[id_] = mask.sum()
            bbox[id_] = np.concatenate([bbox_min, bbox_max])

        return {"ids": instance_ids, "center": centers, "occupancy": occupancy, "bbox": bbox}

    def __getitem__(self, index):
        return_args = self._get_item(index)
        if self.two_stream and (self.phase == DatasetPhase.Train or self.phase == DatasetPhase.TrainVal):
            # apply another data augmentation
            return_args_aux = self._get_item(index)
            for k, v in return_args_aux.items():
                return_args['{}_aux'.format(k)] = v
        return return_args

    def _get_item(self, index):
        inverse_map = None
        coords, feats, labels, idx2hasPoint = self.load_data(index)
        inverse_map = torch.zeros(size=(coords.shape[0], 1))

        # Downsample the pointcloud with finer voxel size before transformation for memory and speed
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            if IS_OLD_ME:
                inds = ME.utils.sparse_quantize(
                    coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
            else:
                _, inds = ME.utils.sparse_quantize(
                    coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)

            inds = np.sort(inds)
            # TODO, some coords will be ignored in ME.SparseTensor
            # if not self.retain_label:
            #     inds = np.sort(inds)
            # else:
            #     # sort is done during merging
            #     inds = t.merge_mapping_index(inds, label_mask=labels != self.ignore_mask)

            # merge inds and inds_valid
            coords = coords[inds]
            feats = feats[inds]
            labels = labels[inds]
            idx2hasPoint = t.update_idx2hasPoint_np(idx2hasPoint, inds, coords, tag='PREVOXELIZATION_VOXEL_SIZE')

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords, feats, labels, idx2hasPoint = self.prevoxel_transform(coords, feats, labels, idx2hasPoint)

        if self.phase == DatasetPhase.Train:
            coords, feats, labels, idx2hasPoint, transformation = self.voxelizer.voxelize(
                coords, feats, labels, idx2hasPoint)
        else:
            coords, feats, labels, idx2hasPoint, transformation, inverse_map = self.voxelizer.voxelize_test(
                coords, feats, labels, idx2hasPoint)

        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, feats, labels, idx2hasPoint = self.input_transform(coords, feats, labels, idx2hasPoint)
        if self.target_transform is not None:
            coords, feats, labels, idx2hasPoint = self.target_transform(coords, feats, labels, idx2hasPoint)

        if self.augment_data:
            # For some networks, making the network invariant to even, odd coords is important
            coords += (torch.rand(3) * 100).int().numpy()

        # ------------- label mapping --------------------
        if self.IGNORE_LABELS is not None:  # TODO speed up
            labels = self.map_label_v2(labels)

        # Use coordinate features if config is set
        if self.AUGMENT_COORDS_TO_FEATS:
            coords, feats, labels, idx2hasPoint = self._augment_coords_to_feats(coords, feats, labels, idx2hasPoint)

        return_args = {'coords': coords, 'feats': feats, 'labels': labels,
                       'idx2hasPoints': idx2hasPoint, 'indexes': index,
                       'inverse_map': inverse_map,
                       # 'instance_labels': instance_labels
                       }
        if self.return_transformation:
            return_args['transformations'] = transformation.astype(np.float32)

        return return_args

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.data_root is not None:
            body.append("Root location: {}".format(self.data_root))
            body.append("Two stream: {}".format(self.two_stream))
        body += self.extra_repr().splitlines()
        if hasattr(self, "input_transform") and self.input_transform is not None:
            body += [repr(self.input_transform)]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return ""


def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           input_transform=None,
                           target_transform=None):
    """
    Return both dataset and dataloader
    """
    if isinstance(phase, str):
        phase = str2datasetphase_type(phase)

    is_train_phase = phase == DatasetPhase.Train or phase == DatasetPhase.TrainVal

    # two_stream only turn on in training
    two_stream = config.DATA.two_stream and is_train_phase
    # retain label only turn on in training
    retain_label = config.DATA.retain_label and is_train_phase
    # special retain label for dropout
    dropout_retain_label = config.DATA.dropout_retain_label and is_train_phase
    # set the point alignment level here
    alignment_level = config.DATA.alignment_level

    if config.DATA.return_transformation:
        collate_fn = t.cflt_collate_fn_factory(limit_numpoints, two_stream=two_stream,
                                               alignment_level=alignment_level,
                                               ignore_label=config.DATA.ignore_label)
    else:
        collate_fn = t.cfl_collate_fn_factory(limit_numpoints, two_stream=two_stream,
                                              alignment_level=alignment_level,
                                              ignore_label=config.DATA.ignore_label)

    prevoxel_transform_train = []
    if augment_data:
        prevoxel_transform_train.append(t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS))

    if len(prevoxel_transform_train) > 0:
        prevoxel_transforms = t.Compose(prevoxel_transform_train)
    else:
        prevoxel_transforms = None

    input_transforms = []
    if input_transform is not None:
        input_transforms += input_transform

    if augment_data:
        input_transforms += [
            t.RandomDropout(0.2, retain_label=dropout_retain_label, ignore_label=config.DATA.ignore_label),
            t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
            # t.ChromaticJitter(config.AUGMENTATION.data_aug_color_jitter_std),
            # t.HueSaturationTranslation(config.AUGMENTATION.data_aug_hue_max, config.AUGMENTATION.data_aug_saturation_max),
        ]

        if config.AUGMENTATION.use_auto_contrast:
            input_transforms.append(
                t.ChromaticAutoContrast(),
            )

        if config.AUGMENTATION.use_translation:
            input_transforms.append(
                t.ChromaticTranslation(config.AUGMENTATION.data_aug_color_trans_ratio),
            )

        if config.AUGMENTATION.use_color_jitter:
            input_transforms.append(
                t.ChromaticJitter(config.AUGMENTATION.data_aug_color_jitter_std)
            )

    if len(input_transforms) > 0:
        input_transforms = t.Compose(input_transforms)
    else:
        input_transforms = None

    dataset = DatasetClass(
        config,
        prevoxel_transform=prevoxel_transforms,
        input_transform=input_transforms,
        target_transform=target_transform,
        cache=config.DATA.cache_data,
        augment_data=augment_data,
        phase=phase)

    data_args = {
        'dataset': dataset,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'collate_fn': collate_fn,
    }

    if repeat:
        if get_world_size() > 1:
            logging.info("Using distributed sampler...")
            data_args['sampler'] = DistributedInfSampler(dataset, shuffle=shuffle)
        else:
            logging.info("Using regular sampler...")
            data_args['sampler'] = InfSampler(dataset, shuffle=shuffle)

    else:
        data_args['shuffle'] = shuffle

    data_loader = DataLoader(**data_args)

    return dataset, data_loader


class DataLoaderForFactory(BaseDataset):
    DATASET_CLASS = None

    def __init__(self, config):
        super().__init__(cfg=config)

        data_cfg = config.DATA

        # dataloader

        DatasetClass = self.DATASET_CLASS.get(data_cfg.dataset)

        logging.info('Initializing dataloader....')

        self.train_dataset, self.train_dataloader = initialize_data_loader(
            DatasetClass, config, phase=data_cfg.train_split,
            num_workers=self.num_workers, augment_data=True,
            shuffle=True, repeat=True, batch_size=self.batch_size,
            limit_numpoints=data_cfg.train_limit_numpoints)

        self.val_dataset, self.val_dataloader = initialize_data_loader(
            DatasetClass, config, phase=data_cfg.val_split,
            num_workers=self.num_val_workers, augment_data=False,
            shuffle=False, repeat=True,
            batch_size=self.val_batch_size, limit_numpoints=False)

        self.test_dataset, self.test_dataloader = initialize_data_loader(
            DatasetClass, config, phase=data_cfg.test_split,
            num_workers=self.num_test_workers, augment_data=False,
            shuffle=False, repeat=True,
            batch_size=self.test_batch_size, limit_numpoints=False)

        self.logger.info("Train dataset: \n {}".format(self.train_dataset))
        self.logger.info("Val dataset: \n {}".format(self.val_dataset))
        self.logger.info("Test dataset: \n {}".format(self.test_dataset))


class ForVisualizeDataLoaderForFactory(BaseDataset):
    DATASET_CLASS = None

    def __init__(self, config):
        super().__init__(cfg=config)

        data_cfg = config.DATA

        # dataloader

        DatasetClass = self.DATASET_CLASS.get(data_cfg.dataset)

        logging.info('Initializing dataloader....')

        self.train_dataset, self.train_dataloader = initialize_data_loader(
            DatasetClass, config, phase=data_cfg.train_split,
            num_workers=self.num_workers, augment_data=False,
            shuffle=False, repeat=True, batch_size=self.batch_size,
            limit_numpoints=data_cfg.train_limit_numpoints)

        self.val_dataset, self.val_dataloader = initialize_data_loader(
            DatasetClass, config, phase=data_cfg.val_split,
            num_workers=self.num_val_workers, augment_data=False,
            shuffle=False, repeat=True,
            batch_size=self.val_batch_size, limit_numpoints=False)

        self.test_dataset, self.test_dataloader = initialize_data_loader(
            DatasetClass, config, phase=data_cfg.test_split,
            num_workers=self.num_test_workers, augment_data=False,
            shuffle=False, repeat=True,
            batch_size=self.test_batch_size, limit_numpoints=False)

        self.logger.info("Train dataset: \n {}".format(self.train_dataset))
        self.logger.info("Val dataset: \n {}".format(self.val_dataset))
        self.logger.info("Test dataset: \n {}".format(self.test_dataset))


def initialize_detection_data_loader(DatasetClass,
                                     config,
                                     phase,
                                     num_workers,
                                     shuffle,
                                     repeat,
                                     augment_data,
                                     batch_size,
                                     use_color,
                                     use_height,
                                     by_points,
                                     by_scenes,
                                     input_transform=None,
                                     target_transform=None):
    is_train_phase = phase == DatasetPhase.Train or phase == DatasetPhase.TrainVal

    def collate_fn(samples):
        data, voxel = [], []
        for sample in samples:
            data.append({w: sample[w] for w in sample if w != 'voxel'})
            voxel.append(sample['voxel'])

        # for non-voxel data, use default collate
        data_batch = default_collate(data)

        batch_ids = np.array(
            [b for b, v in enumerate(voxel) for _ in range(v[0].shape[0])])
        voxel_ids = np.concatenate([v[1] for v in voxel], 0)

        coords = np.concatenate([v[0] for v in voxel], 0)
        coords = np.concatenate([batch_ids[:, None], coords], 1)

        colors = np.concatenate([v[2] for v in voxel], 0)

        data_batch['coords'] = torch.from_numpy(coords).int()
        data_batch['indexes'] = torch.from_numpy(voxel_ids)
        # data_batch['voxel_feats'] = data_batch['point_clouds'].new_ones(batch_ids.shape[0], 3)
        data_batch['feats'] = torch.from_numpy(colors).float()

        return data_batch

    dataset = DatasetClass(
        config,
        phase=phase,
        num_points=20000,
        use_color=use_color,
        use_height=use_height,
        by_scenes=by_points,
        by_points=by_scenes,
        augment=augment_data
    )

    data_args = {
        'dataset': dataset,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'collate_fn': collate_fn,
    }

    if repeat:
        if get_world_size() > 1:
            logging.info("Using distributed sampler...")
            data_args['sampler'] = DistributedInfSampler(dataset, shuffle=shuffle)
        else:
            logging.info("Using regular sampler...")
            data_args['sampler'] = InfSampler(dataset, shuffle=shuffle)
    else:
        data_args['shuffle'] = shuffle

    data_loader = DataLoader(**data_args)

    return dataset, data_loader


class DetectionDataLoaderForFactory(BaseDataset):
    DATASET_CLASS = None

    def __init__(self, config):
        super().__init__(cfg=config)

        data_cfg = config.DATA

        # dataloader
        DatasetClass = self.DATASET_CLASS.get(data_cfg.dataset)

        logging.info('Initializing detection dataloader....')

        # parameters:
        #   by_scenes should be a list including scene_name in scannet
        #   by_points should be a filename which endswith .pth, after loading it, it would be a dict
        #   use_color is a bool
        #   use_height is a bool

        self.train_dataset, self.train_dataloader = initialize_detection_data_loader(
            DatasetClass, config, phase=data_cfg.train_split,
            num_workers=self.num_workers, augment_data=True,
            shuffle=True, repeat=True, batch_size=self.batch_size,
            use_color=False, use_height=False,
            by_points=None, by_scenes=None)

        self.val_dataset, self.val_dataloader = initialize_detection_data_loader(
            DatasetClass, config, phase=data_cfg.val_split,
            num_workers=self.num_val_workers, augment_data=False,
            shuffle=False, repeat=True, batch_size=self.val_batch_size,
            use_color=False, use_height=False,
            by_points=None, by_scenes=None)

        self.test_dataset, self.test_dataloader = initialize_detection_data_loader(
            DatasetClass, config, phase=data_cfg.test_split,
            num_workers=self.num_test_workers, augment_data=False,
            shuffle=False, repeat=True, batch_size=self.test_batch_size,
            use_color=False, use_height=False,
            by_points=None, by_scenes=None)

        self.logger.info("Train dataset: \n {}".format(self.train_dataset))
        self.logger.info("Val dataset: \n {}".format(self.val_dataset))
        self.logger.info("Test dataset: \n {}".format(self.test_dataset))
