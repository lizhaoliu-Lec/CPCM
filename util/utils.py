import logging
import socket
import sys

import glob
import numpy as np
import os

import pandas as pd
import torch
import time

from pyntcloud import PyntCloud

from meta_data.constant import PROJECT_NAME


def get_time_string():
    return time.strftime("%Y%m%d.%H%M%S", time.localtime())


class GPUOccupy:
    def __init__(self):
        self.x = None

    def occupy(self):
        self.x = occupy_mem(os.getenv("CUDA_VISIBLE_DEVICES"))

    def free(self):
        del self.x
        torch.cuda.empty_cache()
        self.x = None


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occupy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.95)
    block_mem = max_mem - used

    if block_mem < 0:
        logger = logging.getLogger(PROJECT_NAME)
        logger.info(
            "Unable to occupy GPU memory, remaining: {}, used: {}, max_mem: {}".format(block_mem, used, max_mem))
        return None
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0

    def update_list(self, val_list):
        for val in val_list:
            self.update(val)


class AverageMeterDict(object):
    def __init__(self, names) -> None:
        self.meter_dict = {name: AverageMeter() for name in names}

    def __iter__(self):
        return iter(self.meter_dict)

    def __getitem__(self, key):
        return self.meter_dict[key]

    def update(self, ret) -> None:
        for k in ret.keys():
            result = ret[k]
            assert k in self.meter_dict
            if type(result) is list:
                self.meter_dict[k].update_list(result)
            else:
                self.meter_dict[k].update(result)

    def update_with_n(self, ret) -> None:
        for k in ret.keys():
            result, n = ret[k]
            assert k in self.meter_dict
            assert type(result) is not list
            self.meter_dict[k].update(result, n)

    def reset(self) -> None:
        for k in self.meter_dict.keys():
            self.meter_dict[k].reset()


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(
        K + 1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(model, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f=''):
    if use_cuda:
        model.cpu()
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d' % epoch + '.pth')
            print('11111', f)
            assert os.path.isfile(f)
        else:
            print(os.path.join(exp_path, exp_name + '*.pth'))
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '*.pth')))
            print('22222', f)
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2: -4])

    if len(f) > 0:
        # logger.info('Restore from ' + f)
        checkpoint = torch.load(f)
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    if use_cuda:
        model.cuda()
    return epoch + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def checkpoint_save(model, exp_path, exp_name, epoch, save_freq=16, use_cuda=True):
    f = os.path.join(exp_path, exp_name + '-%09d' % epoch + '.pth')
    # logger.info('Saving ' + f)
    # model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()

    # remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + '-%09d' % epoch + '.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_model_param(model, pretrained_dict, prefix=""):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items() if
                              k[len_prefix:] in model_dict and prefix in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def print_error(message, user_fault=False):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
        sys.exit(2)
    sys.exit(-1)


def recursively_set_attr(_dict: dict):
    class DynamicField(object):
        pass

        def __repr__(self):
            return '{}'.format(self.__dict__)

    ret = DynamicField()
    for k, v in _dict.items():
        if not isinstance(v, dict):
            setattr(ret, k, v)
        else:
            setattr(ret, k, recursively_set_attr(v))
    return ret


def create_exp_path(config):
    dataset_name = config.DATA.name
    model_name = config.MODEL.name
    time_string = config.log_time

    exp_name = ''
    if hasattr(config.GENERAL, 'exp_name'):
        exp_name = config.GENERAL.exp_name

    host_device = "{}-card{}_".format(socket.gethostname(), os.getenv("CUDA_VISIBLE_DEVICES"))
    exp_name = host_device + exp_name

    exp_path = os.path.join('exp', '{}-{}-{}-{}'.format(dataset_name, model_name, exp_name, time_string))
    config.exp_path = exp_path

    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)

    return exp_path


def euclidean_distance(coords1, coords2):
    """
    coords1: coordinates of the anchor, shape (N, 3)
    coords2: coordinates of the points, shape (M, 3)
    N << 3
    """
    return torch.cdist(coords1.float(), coords2.float(), p=2)  # (N, M)


def recover_color(rgbs):
    return np.array((rgbs + 1) * 127.5, dtype=np.uint8)


def save_ply(coords, rgbs, save_path):
    n = coords.shape[0]
    recovered_color = np.array((rgbs + 1) * 127.5, dtype=np.uint8)

    data = {
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        # 'red': recovered_color[:, 0],
        # 'green': recovered_color[:, 1],
        # 'blue': recovered_color[:, 2],
        'red': np.array(np.random.rand(n) * 255, dtype=np.uint8),
        'blue': np.array(np.random.rand(n) * 255, dtype=np.uint8),
        'green': np.array(np.random.rand(n) * 255, dtype=np.uint8),
    }
    cloud = PyntCloud(pd.DataFrame(data))
    cloud.to_file(save_path)
