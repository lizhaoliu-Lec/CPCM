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

SCANNET_REGISTRY = Registry("ScanNet")
SCANNET_REGISTRY.__doc__ = """
Registry for module that is compatible with scannet dataloader.

The registered object will be called with `obj(cfg)`.
"""


@SCANNET_REGISTRY.register()
class ScannetVoxelizationDataset(VoxelizationDataset):
    # added
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    NUM_IN_CHANNEL = 3
    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    CLASS_LABELS_INSTANCE = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                             'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                             'bathtub', 'otherfurniture']
    VALID_CLASS_IDS_INSTANCE = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    IGNORE_LABELS_INSTANCE = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE))

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    IS_FULL_POINTCLOUD_EVAL = True

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'scannetv2_train.txt',
        DatasetPhase.Val: 'scannetv2_val.txt',
        DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
        DatasetPhase.Test: 'scannetv2_test.txt',
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
        data_root = config.DATA.scannet_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND

        data_paths = read_txt(os.path.join(data_root, 'splits', self.DATA_PATH_FILE[phase]))
        if phase == DatasetPhase.Train and config.DATA.train_file:
            data_paths = read_txt(config.DATA.train_file)

        # data efficiency by sampling points
        self.sampled_inds = {}
        sampled_inds_file = config.DATA.scannet_sampled_inds
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
        coords = pointcloud[0].astype(np.float32)
        feats = pointcloud[1].astype(np.float32)
        if self.phase == DatasetPhase.Test:
            labels = torch.zeros(coords.shape[0]) + self.ignore_mask
            # instance_labels = torch.zeros(coords.shape[0]) + self.ignore_mask
        else:
            labels = pointcloud[2].astype(np.int32)
            # instance_labels = pointcloud[3].astype(np.int32)
            if self.sampled_inds:
                scene_name = self.get_output_id(index)
                mask = np.ones_like(labels).astype(np.bool)
                sampled_inds = self.sampled_inds[scene_name]
                mask[sampled_inds] = False
                labels[mask] = self.ignore_mask  # set it to ignore label
        return coords, feats, labels, np.ones(coords.shape[0])

    def save_features(self, coords, upsampled_features, transformation, iteration, save_dir):
        inds_mapping, xyz = self.get_original_pointcloud(coords, transformation, iteration)
        ptc_feats = upsampled_features.cpu().numpy()[inds_mapping]
        room_id = self.get_output_id(iteration)
        torch.save(ptc_feats, f'{save_dir}/{room_id}')

    def get_original_pointcloud(self, coords, transformation, iteration):
        logging.info('===> Start testing on original pointcloud space.')
        data_path = self.data_paths[iteration]
        fullply_f = self.data_root / data_path
        query_xyz, _, query_label, _ = torch.load(fullply_f)

        coords = coords[:, 1:].numpy() + 0.5
        curr_transformation = transformation[0, :16].numpy().reshape(4, 4)
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
        coords = (np.linalg.inv(curr_transformation) @ coords.T).T

        # Run test for each room.
        from pykeops.numpy import LazyTensor
        # from pykeops.numpy.utils import IsGpuAvailable

        query_xyz = np.array(query_xyz)
        x_i = LazyTensor(query_xyz[:, None, :])  # x_i.shape = (1e6, 1, 3)
        y_j = LazyTensor(coords[:, :3][None, :, :])  # y_j.shape = ( 1, 2e6,3)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
        indKNN = D_ij.argKmin(1, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
        inds = indKNN[:, 0]
        return inds, query_xyz

    def extra_repr(self):
        return "num_classes={}, voxel_size={}".format(len(self.VALID_CLASS_IDS), self.VOXEL_SIZE)


@SCANNET_REGISTRY.register()
class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.02


@SCANNET_REGISTRY.register()
class ScannetVoxelization5cmDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.05


@SCANNET_REGISTRY.register()
class ScannetVoxelization2cmNoElasticDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.02
    ELASTIC_DISTORT_PARAMS = None


@DATASET_REGISTRY.register()
class ScanNetV2DataLoader(DataLoaderForFactory):
    DATASET_CLASS = SCANNET_REGISTRY


@DATASET_REGISTRY.register()
class ForVisualizeScanNetV2DataLoader(ForVisualizeDataLoaderForFactory):
    DATASET_CLASS = SCANNET_REGISTRY


def run_test(config):
    """Test point cloud data loader.
    """
    from torch.utils.data import DataLoader
    from lib.utils import Timer
    import open3d as o3d
    from lib.pc_utils import save_point_cloud

    def make_pcd(coords, feats):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[:, :3].float().numpy())
        pcd.colors = o3d.utility.Vector3dVector(feats[:, :3].numpy() / 255)
        return pcd

    timer = Timer()
    DatasetClass = ScannetVoxelization2cmDataset
    transformations = [
        t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
        t.ChromaticAutoContrast(),
        t.ChromaticTranslation(config.AUGMENTATION.data_aug_color_trans_ratio),
        t.ChromaticJitter(config.AUGMENTATION.data_aug_color_jitter_std),
    ]

    dataset = DatasetClass(
        config,
        prevoxel_transform=t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS),
        input_transform=t.Compose(transformations),
        augment_data=True,
        cache=True,
        elastic_distortion=True)
    logging.info("Train dataset: \n{}".format(dataset))
    exit()
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=t.cfl_collate_fn_factory(limit_numpoints=False),
        batch_size=1,
        shuffle=True)

    # Start from index 1
    save_dir = './tmp'
    os.makedirs(save_dir, exist_ok=True)

    iter = data_loader.__iter__()
    for i in range(100):
        timer.tic()
        coords, feats, labels = iter.next()
        print("===> labels.size(): {}".format(labels.size()))
        print("===> torch.min(feats): {}".format(torch.min(feats)))
        print("===> torch.max(feats): {}".format(torch.max(feats)))
        print("===> torch.mean(feats): {}".format(torch.mean(feats)))
        print("===> coords.size(): {}".format(coords.size()))
        print("===> feats.size(): {}".format(feats.size()))
        print("===> coords[0]: {}".format(coords[0]))
        print("===> coords[1]: {}".format(coords[1]))
        print("===> feats.size(): {}".format(feats.size()))
        pcd = make_pcd(coords, feats)
        # o3d.visualization.draw_geometries([pcd])

        # draw_and_save_point_cloud([coords[:, 1:]], [feats], ['idx={}'.format(i)],
        #                           save_path=os.path.join(save_dir, '{}.html'.format(i)))
        # save_ply(coords[:, 1:], feats, os.path.join(save_dir, '{}.ply'.format(i)))
        # points_3d = np.concatenate([coords[:, 1:].numpy(), feats.numpy(), labels.unsqueeze(-1).numpy()], 1)
        points_3d = np.concatenate([coords[:, 1:].numpy(), feats.numpy()], 1)
        print("===> points_3d.shape: {}".format(points_3d.shape))
        save_point_cloud(points_3d, os.path.join(save_dir, 'scannet_{}.ply'.format(i)), binary=False)

        print(timer.toc())


if __name__ == '__main__':
    from util.config import CfgNode

    cfg = CfgNode(CfgNode.load_yaml_with_base(
        '/home/liulizhao/projects/WeaklySegmentationKit/config/fully_supervised/default.yaml'))

    run_test(cfg)
