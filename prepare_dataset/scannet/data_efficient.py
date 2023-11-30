import os
import shutil
import sys
import plyfile
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

"""
===> avg num points 1473.4725710508922 for percentage 0.01 (1%)
===> avg num points 143.16523463317913 for percentage 0.001 (0.1%)
===> avg num points 18.178453403833444 for percentage 0.0001 (0.001%)
"""

"""
After server deleted version
===> avg num points 1473.4725710508922 for percentage 0.01
===> avg num points 143.16523463317913 for percentage 0.001
===> avg num points 18.178453403833444 for percentage 0.0001
"""


def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open('scannetv2-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        # raw_name = elements[0]
        # nyu40_name = elements[6]
        raw_name = elements[1]
        nyu40_id = elements[4]
        nyu40_name = elements[7]
        raw2scannet[raw_name] = nyu40_id
    return raw2scannet


g_raw2scannet = get_raw2scannet_label_map()
RAW2SCANNET = g_raw2scannet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/cephfs/mixed/dataset/scannet/scans')
    parser.add_argument('--num_instance', default=1, type=int)
    parser.add_argument('--output', default='./output')
    opt = parser.parse_args()
    return opt


def main(config):
    num_instance = config.num_instance

    out_file = os.path.join(config.output, 'points1t{}c'.format(num_instance))
    print("Saving result in {}".format(out_file))
    if os.path.exists(out_file):
        print(out_file, ' exists')
        return

    scene2sampled_ids = {}
    for scene_name in tqdm(os.listdir(config.input)):

        print(scene_name)
        # Over-segmented segments: maps from segment to vertex/point IDs
        segid_to_pointid = {}
        segfile = os.path.join(config.input, scene_name, '{}_vh_clean_2.0.010000.segs.json'.format(scene_name))
        with open(segfile) as jsondata:
            d = json.load(jsondata)
            seg = d['segIndices']
        for i in range(len(seg)):
            if seg[i] not in segid_to_pointid:
                segid_to_pointid[seg[i]] = []
            segid_to_pointid[seg[i]].append(i)

        # Instances over-segmented segment IDs: annotation on segments
        instance_segids = []
        annotation_filename = os.path.join(config.input, scene_name, '{}.aggregation.json'.format(scene_name))
        with open(annotation_filename) as jsondata:
            d = json.load(jsondata)
            for x in d['segGroups']:
                instance_segids.append(x['segments'])

        # Each instance's points
        sampled_ids = []
        for i in range(len(instance_segids)):
            segids = instance_segids[i]
            pointids = []
            for segid in segids:
                pointids += segid_to_pointid[segid]
            pointids = np.array(pointids)
            sampled_ids.extend(list(pointids[:num_instance]))

        scene2sampled_ids[scene_name] = np.array(sampled_ids)

    torch.save(scene2sampled_ids, out_file)


def generate_random_ids(high, num):
    assert num <= high, 'num={}, high={}'.format(num, high)
    a = np.arange(high)
    np.random.shuffle(a)
    return a[:num]


def get_pth_label(filepath):
    return torch.load(filepath)[2].astype(np.int32)


def main_for_efficient_pts():
    cfg = parse_args()
    print("===> making output dir: {}".format(os.path.abspath(cfg.output)))
    os.makedirs(cfg.output, exist_ok=True)
    main(cfg)


def main_for_efficient_percentage():
    root_dir = "/home/liulizhao/datasets/scannet_fully_supervised_preprocessed"
    save_dir = "/home/liulizhao/datasets/scannet_fully_supervised_preprocessed/points"
    for perc in [0.01, 0.001, 0.0001]:  # 1%, 0.1%, 0.01%
        main_for_efficient_percentage_helper(
            root_dir=root_dir,
            save_dir=save_dir,
            percentage=perc,
        )


def main_for_efficient_percentage_helper(root_dir, save_dir, percentage):
    save_path = os.path.join(save_dir, 'percentage{}evenc'.format(percentage))

    # check for already exists
    if os.path.exists(save_path):
        return

    points_inds = {}
    recorded_num_points = []
    for scene_name in tqdm(os.listdir(root_dir)):
        scene_name = scene_name.split('.')[0]  # remove .pth
        if not 'scene' in scene_name:  # skip the points, splits, save_dir folder
            continue
        assert scene_name not in points_inds
        scene_file = os.path.join(root_dir, scene_name + '.pth')
        if not os.path.exists(scene_file):
            raise ValueError("Can not find file {}".format(scene_file))

        labels = get_pth_label(scene_file)

        unique_labels = np.unique(labels)
        _point_ids = []

        # label perc points for each class
        for unique_label in unique_labels:
            unique_label_ids = np.where(labels == unique_label)[0]
            num_points_class = len(unique_label_ids)
            if num_points_class < 1:
                continue
            num_sampled = max(int(num_points_class * percentage), 1)
            _point_ids.extend(generate_random_ids(num_points_class, num_sampled))

        points_inds[scene_name] = np.array(_point_ids)
        recorded_num_points.append(points_inds[scene_name].shape[0])

    torch.save(points_inds, save_path)
    print("===> avg num points {} for percentage {}".format(np.mean(recorded_num_points), percentage))


if __name__ == '__main__':
    # main_for_efficient_pts()
    main_for_efficient_percentage()
