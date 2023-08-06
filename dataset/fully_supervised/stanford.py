import logging
import os
import sys
import numpy as np
from collections import defaultdict

from fvcore.common.registry import Registry
from scipy import spatial
import torch
from plyfile import PlyData

from ..build import DATASET_REGISTRY
from lib.utils import read_txt, fast_hist, per_class_iu
from dataset.fully_supervised.base import VoxelizationDataset, DatasetPhase, str2datasetphase_type, cache, \
    DataLoaderForFactory, ForVisualizeDataLoaderForFactory
import dataset.fully_supervised.transforms as t

STANFORD_REGISTRY = Registry("Stanford")
STANFORD_REGISTRY.__doc__ = """
Registry for module that is compatible with stanford dataloader.

The registered object will be called with `obj(cfg)`.
"""


class StanfordVoxelizationDatasetBase:
    # added
    NUM_LABELS = 14
    CLASS_LABELS = ('clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
                    'door', 'floor', 'sofa', 'table', 'wall', 'window')
    VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13)
    IGNORE_LABELS = tuple(set(range(14)) - set(VALID_CLASS_IDS))

    CLASS_LABELS_INSTANCE = ('clutter', 'beam', 'board', 'bookcase', 'chair', 'column',
                             'door', 'sofa', 'table', 'window')
    VALID_CLASS_IDS_INSTANCE = (0, 1, 2, 3, 5, 6, 7, 9, 11, 13)
    IGNORE_LABELS_INSTANCE = tuple(set(range(14)) - set(VALID_CLASS_IDS_INSTANCE))

    # ---------

    CLIP_SIZE = None
    CLIP_BOUND = None
    LOCFEAT_IDX = 2
    ROTATION_AXIS = 'z'
    # IGNORE_LABELS = (10,)  # remove stairs, following SegCloud

    # CLASSES = [
    #     'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
    #     'table', 'wall', 'window'
    # ]

    IS_FULL_POINTCLOUD_EVAL = True

    DATA_PATH_FILE = {
        DatasetPhase.Train: 'train.txt',
        DatasetPhase.Val: 'val.txt',
        DatasetPhase.TrainVal: 'trainval.txt',
        DatasetPhase.Test: 'test.txt'
    }

    def test_pointcloud(self, pred_dir):
        print('Running full pointcloud evaluation.')
        # Join room by their area and room id.
        room_dict = defaultdict(list)
        for i, data_path in enumerate(self.data_paths):
            area, room = data_path.split(os.sep)
            room, _ = os.path.splitext(room)
            room_id = '_'.join(room.split('_')[:-1])
            room_dict[(area, room_id)].append(i)
        # Test independently for each room.
        sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
        pred_list = sorted(os.listdir(pred_dir))
        hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
        for room_idx, room_list in enumerate(room_dict.values()):
            print(f'Evaluating room {room_idx} / {len(room_dict)}.')
            # Join all predictions and query pointclouds of split data.
            pred = np.zeros((0, 4))
            pointcloud = np.zeros((0, 7))
            for i in room_list:
                pred = np.vstack((pred, np.load(os.path.join(pred_dir, pred_list[i]))))
                pointcloud = np.vstack((pointcloud, self.load_ply(i)[0]))
            # Deduplicate all query pointclouds of split data.
            pointcloud = np.array(list(set(tuple(l) for l in pointcloud.tolist())))
            # Run test for each room.
            pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
            _, result = pred_tree.query(pointcloud[:, :3])
            ptc_pred = pred[result, 3].astype(int)
            ptc_gt = pointcloud[:, -1].astype(int)
            if self.IGNORE_LABELS:
                ptc_pred = self.label2masked[ptc_pred]
                ptc_gt = self.label2masked[ptc_gt]
            hist += fast_hist(ptc_pred, ptc_gt, self.NUM_LABELS)
            # Print results.
            ious = []
            print('Per class IoU:')
            for i, iou in enumerate(per_class_iu(hist) * 100):
                result_str = ''
                if hist.sum(1)[i]:
                    result_str += f'{iou}'
                    ious.append(iou)
                else:
                    result_str += 'N/A'  # Do not print if data not in ground truth.
                print(result_str)
            print(f'Average IoU: {np.nanmean(ious)}')

    def _augment_coords_to_feats(self, coords, feats, labels=None):
        # Center x,y
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = coords - coords_center
        feats = np.concatenate((feats, norm_coords), 1)
        return coords, feats, labels


@STANFORD_REGISTRY.register()
class StanfordDataset(StanfordVoxelizationDatasetBase, VoxelizationDataset):
    # Voxelization arguments
    VOXEL_SIZE = 0.05  # 5cm

    CLIP_BOUND = 4  # [-N, N]
    TEST_CLIP_BOUND = None

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = \
        ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

    # AUGMENT_COORDS_TO_FEATS = True
    # NUM_IN_CHANNEL = 6
    AUGMENT_COORDS_TO_FEATS = False
    NUM_IN_CHANNEL = 3

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 augment_data=True,
                 elastic_distortion=False,
                 phase=DatasetPhase.Train):
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        data_root = config.DATA.stanford3d_path
        if isinstance(self.DATA_PATH_FILE[phase], (list, tuple)):
            data_paths = []
            for split in self.DATA_PATH_FILE[phase]:
                data_paths += read_txt(os.path.join(data_root, 'splits', split))
        else:
            data_paths = read_txt(os.path.join(data_root, 'splits', self.DATA_PATH_FILE[phase]))

        if config.DATA.voxel_size:
            self.VOXEL_SIZE = config.DATA.voxel_size

        # data efficiency by sampling points
        self.sampled_inds = {}
        sampled_inds_file = config.DATA.stanford3d_sampled_inds
        if sampled_inds_file and phase == DatasetPhase.Train:
            logging.info("Loading sampled_ids from {}".format(sampled_inds_file))
            self.sampled_inds = torch.load(sampled_inds_file)

        logging.info('voxel size: {}'.format(self.VOXEL_SIZE))
        logging.info('Loading {} {}: {}'.format(self.__class__.__name__, phase,
                                                self.DATA_PATH_FILE[phase]))

        VoxelizationDataset.__init__(
            self,
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

    # @cache # comment out to support two stream data loading
    def load_ply(self, index):
        filepath = self.data_root / self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        return coords, feats, labels, None

    def get_output_id(self, index):
        return self.data_paths[index]

    # @cache # comment out to support two stream data loading
    def load_data(self, index):
        filepath = self.data_root / self.data_paths[index]
        pointcloud = torch.load(filepath)
        coords = pointcloud[:, :3].astype(np.float32)
        feats = pointcloud[:, 3:6].astype(np.float32)
        labels = pointcloud[:, 6].astype(np.int32)

        if self.sampled_inds:
            scene_name = self.get_output_id(index)
            mask = np.ones_like(labels).astype(np.bool)
            sampled_inds = self.sampled_inds[scene_name]
            mask[sampled_inds] = False
            labels[mask] = self.ignore_mask  # set it to ignore label

        return coords, feats, labels, np.ones(coords.shape[0])

    def extra_repr(self):
        return "num_classes={}, voxel_size={}".format(len(self.VALID_CLASS_IDS), self.VOXEL_SIZE)


@STANFORD_REGISTRY.register()
class StanfordArea5Dataset(StanfordDataset):
    DATA_PATH_FILE = {
        DatasetPhase.Train: ['area1.txt', 'area2.txt', 'area3.txt', 'area4.txt', 'area6.txt'],
        DatasetPhase.Val: 'area5.txt',
        DatasetPhase.Test: 'area5.txt',
    }


@STANFORD_REGISTRY.register()
class StanfordArea52cmDataset(StanfordArea5Dataset):
    VOXEL_SIZE = 0.02


@STANFORD_REGISTRY.register()
class StanfordArea53cmDataset(StanfordArea5Dataset):
    CLIP_BOUND = 3.2
    VOXEL_SIZE = 0.03


@STANFORD_REGISTRY.register()
class StanfordArea57d5cmDataset(StanfordArea5Dataset):
    VOXEL_SIZE = 0.075


@STANFORD_REGISTRY.register()
class StanfordArea510cmDataset(StanfordArea5Dataset):
    VOXEL_SIZE = 0.1


@DATASET_REGISTRY.register()
class StanfordDataLoader(DataLoaderForFactory):
    DATASET_CLASS = STANFORD_REGISTRY


@DATASET_REGISTRY.register()
class ForVisualizeStanfordDataLoader(ForVisualizeDataLoaderForFactory):
    DATASET_CLASS = STANFORD_REGISTRY


def run_test(config):
    """Test point cloud data loader.
    """
    from torch.utils.data import DataLoader
    from lib.utils import Timer
    import open3d as o3d
    from util.utils import save_ply
    from util.visualize import draw_and_save_point_cloud
    from lib.pc_utils import save_point_cloud

    def make_pcd(coords, feats):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[:, :3].float().numpy())
        pcd.colors = o3d.utility.Vector3dVector(feats[:, :3].numpy() / 255)
        return pcd

    timer = Timer()
    DatasetClass = StanfordArea5Dataset
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
        points_3d = np.concatenate([coords[:, 1:].numpy(), feats.numpy(), labels.unsqueeze(-1).numpy()], 1)
        print("===> points_3d.shape: {}".format(points_3d.shape))
        save_point_cloud(points_3d, os.path.join(save_dir, '{}.ply'.format(i)), with_label=True)

        print(timer.toc())


if __name__ == '__main__':
    from util.config import CfgNode

    cfg = CfgNode(CfgNode.load_yaml_with_base(
        '/home/liulizhao/projects/WeaklySegmentationKit/config/fully_supervised/default.yaml'))

    run_test(cfg)
