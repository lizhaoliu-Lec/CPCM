import os.path

import torch
import numpy as np
import tqdm

np.random.seed(123)  # fix random seed

"""
First few random id when num=2
===> generate_random_ids: [286316 409108]
===> generate_random_ids: [ 9758 11513]
===> generate_random_ids: [444701  39776]
===> generate_random_ids: [225578 927480]
===> generate_random_ids: [192746 409704]
===> generate_random_ids: [120466 382904]
===> generate_random_ids: [149828  26952]
===> generate_random_ids: [ 34966 153102]
===> generate_random_ids: [398814 374657]
"""

"""
NUM AVG POINTS for percentage
===> avg num points 1281.577205882353 for percentage 0.002
===> avg num points 64103.19117647059 for percentage 0.1
===> avg num points 127.67647058823529 for percentage 0.0002
"""

"""
NUM AVG POINTS for CLASS wise percentage
===> avg num points 16666.45588235294 for percentage 0.002
===> avg num points 1666.2058823529412 for percentage 0.0002
"""

"""
NUM AVG POINTS for class even percentage
===> avg num points 1243.5220588235295 for percentage 0.002 (0.2%)
===> avg num points 62385.330882352944 for percentage 0.1 (10%)
===> avg num points 120.65808823529412 for percentage 0.0002 (0.02%)

NUM AVG POINTS for class even percentage settings from SQN
===> avg num points 619.6323529411765 for percentage 0.001 (0.1%)
===> avg num points 58.91544117647059 for percentage 0.0001
"""

"""
===> after server deleted
===> avg num points 619.6323529411765 for percentage 0.001 (0.1%)
===> avg num points 58.91544117647059 for percentage 0.0001 (0.01%)
===> avg num points 6234.676470588235 for percentage 0.01 (1%)
"""

NUM_CLASS = 13


def read_txt(filepath):
    with open(filepath) as f:
        return [_.strip() for _ in f.readlines()]


def load_pth(filepath):
    pointcloud = torch.load(filepath)
    coords = pointcloud[:, :3].astype(np.float32)
    feats = pointcloud[:, 3:6].astype(np.float32)
    labels = pointcloud[:, 6].astype(np.int32)
    return coords, feats, labels


def get_pth_size(filepath):
    return load_pth(filepath)[0].shape[0]


def get_pth_label(filepath):
    return load_pth(filepath)[2]


def generate_random_ids(high, num):
    assert num <= high, 'num={}, high={}'.format(num, high)
    a = np.arange(high)
    np.random.shuffle(a)
    return a[:num]


def data_efficient_by_num(save_dir, path2points, num):
    points_inds = {}
    for pth_path in tqdm.tqdm(path2points.keys()):
        # print("===> pth_path: {}".format(pth_path))
        num_points = path2points[pth_path]
        # print("===> num_points: {}".format(num_points))
        # print("===> generate_random_ids: {}".format(generate_random_ids(num_points, 2)))
        assert pth_path not in points_inds
        points_inds[pth_path] = generate_random_ids(num_points, num)

    torch.save(points_inds, os.path.join(save_dir, 'points{}'.format(num)))


def data_efficient_by_nums(base_dir, save_dir, nums):
    os.makedirs(save_dir, exist_ok=True)
    txt_paths = [
        os.path.join(base_dir, 'splits/area{}.txt'.format(_)) for _ in range(1, 7)
    ]

    all_pth_files = []
    for txt_path in txt_paths:
        all_pth_files.extend(read_txt(txt_path))

    pth_size = {}
    for pth_path in tqdm.tqdm(all_pth_files):
        # print("===> pth_path: {}".format(pth_path))
        num_points = get_pth_size(os.path.join(base_dir, pth_path))

        assert pth_path not in pth_size
        pth_size[pth_path] = num_points

    for num in nums:
        data_efficient_by_num(save_dir, pth_size, num)


def data_efficient_by_percentage_v2(base_dir, save_dir, all_pth_files, percentage):
    save_path = os.path.join(save_dir, 'percentage{}evenc'.format(percentage))
    if os.path.exists(save_path):
        return
    points_inds = {}
    recorded_num_points = []

    for pth_path in tqdm.tqdm(all_pth_files):
        # print("===> pth_path: {}".format(pth_path))
        # num_points = get_pth_size(os.path.join(base_dir, pth_path))
        labels = get_pth_label(os.path.join(base_dir, pth_path))
        _point_ids = []

        # class-wise even sampling, according to PSD ICCV 2021.
        # TODO bug here, no label for class 13
        for label in range(NUM_CLASS):
            unique_label_ids = np.where(labels == label)[0]
            num_points_class = len(unique_label_ids)
            if num_points_class < 1:
                continue
            num_sampled = max(int(num_points_class * percentage), 1)
            _point_ids.extend(generate_random_ids(num_points_class, num_sampled).tolist())

        assert pth_path not in points_inds
        points_inds[pth_path] = np.array(_point_ids)
        recorded_num_points.append(points_inds[pth_path].shape[0])

    torch.save(points_inds, save_path)
    print("===> avg num points {} for percentage {}".format(np.mean(recorded_num_points), percentage))


def data_efficient_by_percentage_v3(base_dir, save_dir, all_pth_files, percentage):
    save_path = os.path.join(save_dir, 'percentage{}evencv3'.format(percentage))
    if os.path.exists(save_path):
        return
    points_inds = {}
    recorded_num_points = []

    for pth_path in tqdm.tqdm(all_pth_files):
        # print("===> pth_path: {}".format(pth_path))
        # num_points = get_pth_size(os.path.join(base_dir, pth_path))
        labels = get_pth_label(os.path.join(base_dir, pth_path))
        _point_ids = []

        # class-wise even sampling, according to PSD ICCV 2021.
        for label in np.unique(labels):
            unique_label_ids = np.where(labels == label)[0]
            num_points_class = len(unique_label_ids)
            if num_points_class < 1:
                continue
            num_sampled = max(int(num_points_class * percentage), 1)
            _point_ids.extend(generate_random_ids(num_points_class, num_sampled).tolist())

        assert pth_path not in points_inds
        points_inds[pth_path] = np.array(_point_ids)
        recorded_num_points.append(points_inds[pth_path].shape[0])

    torch.save(points_inds, save_path)
    print("===> avg num points {} for percentage {}".format(np.mean(recorded_num_points), percentage))


def data_efficient_by_percentages(base_dir, save_dir, percentages, version=2):
    assert version in [2, 3]
    os.makedirs(save_dir, exist_ok=True)
    txt_paths = [
        os.path.join(base_dir, 'splits/area{}.txt'.format(_)) for _ in range(1, 7)
    ]

    all_pth_files = []
    for txt_path in txt_paths:
        all_pth_files.extend(read_txt(txt_path))

    _api = None
    if version == 2:
        _api = data_efficient_by_percentage_v2
    if version == 3:
        _api = data_efficient_by_percentage_v3
    assert _api is not None

    for percentage in percentages:
        _api(base_dir, save_dir, all_pth_files, percentage)


def generate_for_mil_settings():
    NUM_POINTS = [20, 50, 100, 200]
    PERCENTAGES = [0.002, 0.1, 0.0002]
    base_dir = '/mnt/cephfs/mixed/dataset/stanford_fully_supervised_preprocessed'
    save_dir = '/mnt/cephfs/mixed/dataset/stanford_fully_supervised_preprocessed/points'
    # data_efficient_by_nums(base_dir, save_dir, NUM_POINTS)
    data_efficient_by_percentages(base_dir, save_dir, PERCENTAGES)


def generate_for_sqn_settings():
    PERCENTAGES = [0.001, 0.0001]
    base_dir = '/home/liulizhao/datasets/stanford_fully_supervised_preprocessed'
    save_dir = '/home/liulizhao/datasets/stanford_fully_supervised_preprocessed/points'
    data_efficient_by_percentages(base_dir, save_dir, PERCENTAGES)


def generate_for_more_annotation_settings():
    PERCENTAGES = [0.01]  # only 1%, 10% already generated
    base_dir = '/home/liulizhao/datasets/stanford_fully_supervised_preprocessed'
    save_dir = '/home/liulizhao/datasets/stanford_fully_supervised_preprocessed/points'
    # data_efficient_by_nums(base_dir, save_dir, NUM_POINTS)
    data_efficient_by_percentages(base_dir, save_dir, PERCENTAGES)


if __name__ == '__main__':
    # generate_for_mil_settings()
    # generate_for_sqn_settings()
    generate_for_more_annotation_settings()
