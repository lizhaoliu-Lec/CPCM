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
    parser.add_argument('--output', default='./output')
    parser.add_argument('--splits', default='train')
    opt = parser.parse_args()
    return opt


def main(config):
    if config.split == 'test':
        config.input = ''.join([config.input, '_test'])

    for scene_name in tqdm(os.listdir(config.input)):
        out_file = os.path.join(config.output, scene_name + '.pth')
        if os.path.exists(out_file):
            print(out_file, ' exists')
            continue

        print(scene_name)
        # Raw points in XYZRGBA
        ply_filename = os.path.join(config.input, scene_name, '{}_vh_clean_2.ply'.format(scene_name))
        f = plyfile.PlyData().read(ply_filename)
        points = np.array([list(x) for x in f.elements[0]])

        colors = points[:, 3:6]
        points = points[:, 0:3]  # XYZ+RGB+NORMAL

        if config.split == "train":
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
            labels = []
            annotation_filename = os.path.join(config.input, scene_name, '{}.aggregation.json'.format(scene_name))
            with open(annotation_filename) as jsondata:
                d = json.load(jsondata)
                for x in d['segGroups']:
                    instance_segids.append(x['segments'])
                    labels.append(x['label'])

            # Each instance's points
            instance_labels = np.zeros(points.shape[0])
            semantic_labels = np.zeros(points.shape[0])
            for i in range(len(instance_segids)):
                segids = instance_segids[i]
                pointids = []
                for segid in segids:
                    pointids += segid_to_pointid[segid]
                pointids = np.array(pointids)
                instance_labels[pointids] = i + 1
                semantic_labels[pointids] = RAW2SCANNET[labels[i]]
            torch.save((points, colors, semantic_labels, instance_labels), out_file)
        else:
            torch.save((points, colors), out_file)


if __name__ == '__main__':
    cfg = parse_args()
    print("===> making output dir: {}".format(os.path.abspath(cfg.output)))
    os.makedirs(cfg.output, exist_ok=True)
    main(cfg)
    # # cp splits to output path
    # split_output = os.path.join(cfg.output, 'splits')
    # os.makedirs(split_output, exist_ok=True)
    # split_input = './splits'
    # for split_file in os.listdir(split_input):
    #     shutil.copy(os.path.join(split_input, split_file), os.path.join(split_output, split_file))
