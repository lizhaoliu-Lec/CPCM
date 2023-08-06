import argparse
import math
import os
from operator import itemgetter

from plotly.graph_objs.layout import scene
from plyfile import PlyData
from plotly.subplots import make_subplots
from plotly import graph_objects
from pyntcloud import PyntCloud

import numpy as np
import torch

from meta_data.scan_net import SEMANTIC_NAMES, CLASS_COLOR, COLOR20, SEMANTIC_IDX2NAME


def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.room_split, opt.room_name + '_inst_nostuff.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    if opt.room_split == 'test':
        xyz, rgb = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if opt.task == 'semantic_gt':
        assert opt.room_split != 'test'
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif opt.task == 'instance_gt':
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif opt.task == 'semantic_pred':
        assert opt.room_split != 'train'
        semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif opt.task == 'instance_pred':
        assert opt.room_split != 'train'
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if float(masks[i][2]) < 0.09:
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    if opt.room_split != 'test':
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


"""
Reference: https://github.com/kumuji/mix3d/blob/master/mix3d/utils/point_cloud_utils.py
"""


def old_load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array([data["red"], data["green"], data["blue"]], dtype=np.uint8).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.int32)
    return coords, feats, labels


def load_ply(filepath):
    plydata = PyntCloud.from_file(filepath)

    data = plydata.points
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    if ({"red", "green", "blue"} - set(data.columns)) == set():
        feats = np.array([data["red"], data["green"], data["blue"]], dtype=np.uint8).T
    labels = None
    if "label" in data.columns:
        labels = np.array(data["label"], dtype=np.int32)
    return coords, feats, labels


"""
Reference: https://github.com/kumuji/mix3d/blob/master/mix3d/utils/pc_visualizations.py
"""


def draw_point_cloud(coords, colors=None, label_text=None, legend=None):
    marker = dict(size=1, opacity=0.8)
    if colors is not None:
        marker.update({"color": colors})
    if (colors is None) and (label_text is not None):
        marker.update({"color": label_text})

    point_cloud_image = graph_objects.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        text=label_text,
        mode="markers",
        marker=marker,
        name=legend,
    )
    return point_cloud_image


def draw_and_save_point_cloud(coords_list, colors_list=None,
                              title_list=None, label_text_list=None,
                              save_path=None, verbose=False, n_cols=2):
    assert save_path is not None, 'save_path must not be None'

    num_plot = len(coords_list)

    num_row = math.ceil(num_plot / n_cols)

    specs = [[{"type": "scatter3d"} for j in range(n_cols)] for i in range(num_row)]

    fig = make_subplots(rows=num_row, cols=n_cols, specs=specs)

    for index in range(num_plot):
        coords = coords_list[index]
        colors = None if not colors_list else colors_list[index]
        label_text = None if not label_text_list else label_text_list[index]
        title = None if not title_list else title_list[index]  # TODO use title

        point_cloud_image = draw_point_cloud(coords, colors, label_text, legend=title)

        # get row
        row = index // n_cols
        col = index % n_cols

        # print("===> index: {}, row: {}, col: {}".format(index, row, col))

        fig.add_trace(point_cloud_image, row=row + 1, col=col + 1)

    fig.update_layout(xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=False))

    if verbose:
        print("===> Saving html to path: {}...".format(os.path.abspath(save_path)))

    fig.write_html(save_path)

    if verbose:
        print("===> Saving html to path: {} done".format(os.path.abspath(save_path)))


def draw_point_cloud_plolty(coordinates, pred, target, figure_info, path, name):
    pred_color, pred_text = np.zeros((len(pred), 3)), np.full((len(pred)), "empty")
    target_color, target_text = np.zeros((len(target), 3)), np.full((len(target)), "empty")

    for k, v in figure_info.items():
        pred_color[pred == k] = v["color"]
        pred_text[pred == k] = v["name"]
        target_color[target == k] = v["color"]
        target_text[target == k] = v["name"]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])
    fig.add_trace(draw_point_cloud(coordinates, pred_color, pred_text), row=1, col=1)
    fig.add_trace(draw_point_cloud(coordinates, target_color, target_text), row=1, col=2)

    path = os.path.join(path, name)
    # fig.write_html(path + 'html')
    fig.write_image(path + '.jpeg')


def main():
    pass


if __name__ == '__main__':
    main()
