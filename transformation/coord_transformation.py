import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import convolve
import random
import math


def jitter_flip_rotate(x, jitter=False, flip=False, rotate=False):
    if jitter or flip or rotate:
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rotate:
            theta = random.random() * 2 * math.pi
            # print ('rand',theta)
            m = np.matmul(m, [
                [math.cos(theta), math.sin(theta), 0],
                [-math.sin(theta), math.cos(theta), 0],
                [0, 0, 1]
            ])  # rotation
        return np.matmul(x, m)
    else:
        return x


class JitterFlipRotate:
    def __init__(self, jitter=False, flip=False, rotate=False):
        self.jitter = jitter
        self.flip = flip
        self.rotate = rotate

    def __call__(self, x):
        return jitter_flip_rotate(x, self.jitter, self.flip, self.rotate)


def scale(x, scale_factor):
    return x * scale_factor


class Scale:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, x):
        return scale(x, self.scale_factor)


def elastic(x, gran, mag):
    """
    Elastic distortion
    """
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32) // gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


class Elastic:
    def __init__(self, gran, mag):
        self.gran = gran
        self.mag = mag

    def __call__(self, x):
        return elastic(x, self.gran, self.mag)


def subtract_min(x):
    x -= x.min(0)
    return x


class SubtractMin:
    def __call__(self, x):
        return subtract_min(x)


def crop(xyz, size, max_npoint):
    """
    :param xyz: (n, 3) >= 0
    :param size: (h, w)
    :param max_npoint: max num point
    """
    assert len(size) == 2, 'scale should be in (h, w) format, bug got `{}`'.format(scale)
    h, w = size
    larger_side = max(h, w)
    xyz_offset = xyz.copy()
    valid_idxs = (xyz_offset.min(1) >= 0)
    assert valid_idxs.sum() == xyz.shape[0]

    full_size = np.array([w] * 3)  # TODO, why w? large side of the scale?
    room_range = xyz.max(0) - xyz.min(0)
    while valid_idxs.sum() > max_npoint:
        offset = np.clip(full_size - room_range + 0.001, None, 0) * np.random.rand(3)
        xyz_offset = xyz + offset
        valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_size).sum(1) == 3)
        full_size[:2] -= 32

    return xyz_offset, valid_idxs


class Crop:
    def __init__(self, size, max_npoint):
        self.size = size
        self.max_npoint = max_npoint

    def __call__(self, x):
        return crop(x, self.size, self.max_npoint)
