# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement
import pandas as pd

COLOR_MAP_RGB = (
    (241, 255, 82),
    (102, 168, 226),
    (0, 255, 0),
    (113, 143, 65),
    (89, 173, 163),
    (254, 158, 137),
    (190, 123, 75),
    (100, 22, 116),
    (0, 18, 141),
    (84, 84, 84),
    (85, 116, 127),
    (255, 31, 33),
    (228, 228, 228),
    (0, 255, 0),
    (70, 145, 150),
    (237, 239, 94),
)
IGNORE_COLOR = (0, 0, 0)


def read_plyfile(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
    """Save an RGB point cloud as a PLY file.

    Args:
      points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
          the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
    """
    assert points_3d.ndim == 2
    if with_label:
        assert points_3d.shape[1] == 7
        python_types = (float, float, float, int, int, int, int)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                     ('blue', 'u1'), ('label', 'u1')]
    else:
        if points_3d.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
            points_3d = np.hstack((points_3d, gray_concat))
        assert points_3d.shape[1] == 6
        python_types = (float, float, float, int, int, int)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                     ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, 'w') as f:
            f.write('ply\n'
                    'format ascii 1.0\n'
                    'element vertex %d\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'property uchar alpha\n'
                    'end_header\n' % points_3d.shape[0])
            for row_idx in range(points_3d.shape[0]):
                X, Y, Z, R, G, B = points_3d[row_idx]
                f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
    if verbose is True:
        print('Saved point cloud to: %s' % filename)


class Camera(object):

    def __init__(self, intrinsics):
        self._intrinsics = intrinsics
        self._camera_matrix = self.build_camera_matrix(self.intrinsics)
        self._K_inv = inv(self.camera_matrix)

    @staticmethod
    def build_camera_matrix(intrinsics):
        """Build the 3x3 camera matrix K using the given intrinsics.

        Equation 6.10 from HZ.
        """
        f = intrinsics['focal_length']
        pp_x = intrinsics['pp_x']
        pp_y = intrinsics['pp_y']

        K = np.array([[f, 0, pp_x], [0, f, pp_y], [0, 0, 1]], dtype=np.float32)
        # K[:, 0] *= -1.  # Step 1 of Kyle
        assert matrix_rank(K) == 3
        return K

    @staticmethod
    def extrinsics2RT(extrinsics):
        """Convert extrinsics matrix to separate rotation matrix R and translation vector T.
        """
        assert extrinsics.shape == (4, 4)
        R = extrinsics[:3, :3]
        T = extrinsics[3, :3]
        R = np.copy(R)
        T = np.copy(T)
        T = T.reshape(3, 1)
        R[0, :] *= -1.  # Step 1 of Kyle
        T *= 100.  # Convert from m to cm
        return R, T

    def project(self, points_3d, extrinsics=None):
        """Project a 3D point in camera coordinates into the camera/image plane.

        Args:
          point_3d:
        """
        if extrinsics is not None:  # Map points to camera coordinates
            points_3d = self.world2camera(extrinsics, points_3d)

        # TODO: Make sure to handle homogeneous AND non-homogeneous coordinate points
        # TODO: Consider handling a set of points
        raise NotImplementedError

    def backproject(self,
                    depth_map,
                    labels=None,
                    max_depth=None,
                    max_height=None,
                    min_height=None,
                    rgb_img=None,
                    extrinsics=None,
                    prune=True):
        """Backproject a depth map into 3D points (camera coordinate system). Attach color if RGB image
        is provided, otherwise use gray [128 128 128] color.

        Does not show points at Z = 0 or maximum Z = 65535 depth.

        Args:
          labels: Tensor with the same shape as depth map (but can be 1-channel or 3-channel).
          max_depth: Maximum depth in cm. All pts with depth greater than max_depth will be ignored.
          max_height: Maximum height in cm. All pts with height greater than max_height will be ignored.

        Returns:
          points_3d: Numpy array of size Nx3 (XYZ) or Nx6 (XYZRGB).
        """
        if labels is not None:
            assert depth_map.shape[:2] == labels.shape[:2]
            if (labels.ndim == 2) or ((labels.ndim == 3) and (labels.shape[2] == 1)):
                n_label_channels = 1
            elif (labels.ndim == 3) and (labels.shape[2] == 3):
                n_label_channels = 3

        if rgb_img is not None:
            assert depth_map.shape[:2] == rgb_img.shape[:2]
        else:
            rgb_img = np.ones_like(depth_map, dtype=np.uint8) * 255

        # Convert from 1-channel to 3-channel
        if (rgb_img.ndim == 3) and (rgb_img.shape[2] == 1):
            rgb_img = np.tile(rgb_img, [1, 1, 3])

        # Convert depth map to single channel if it is multichannel
        if (depth_map.ndim == 3) and depth_map.shape[2] == 3:
            depth_map = np.squeeze(depth_map[:, :, 0])
        depth_map = depth_map.astype(np.float32)

        # Get image dimensions
        H, W = depth_map.shape

        # Create meshgrid (pixel coordinates)
        Z = depth_map
        A, B = np.meshgrid(range(W), range(H))
        ones = np.ones_like(A)
        grid = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis], ones[:, :, np.newaxis]),
                              axis=2)
        grid = grid.astype(np.float32) * Z[:, :, np.newaxis]
        # Nx3 where each row is (a*Z, b*Z, Z)
        grid_flattened = grid.reshape((-1, 3))
        grid_flattened = grid_flattened.T  # 3xN where each col is (a*Z, b*Z, Z)
        prod = np.dot(self.K_inv, grid_flattened)
        XYZ = np.concatenate((prod[:2, :].T, Z.flatten()[:, np.newaxis]), axis=1)  # Nx3
        XYZRGB = np.hstack((XYZ, rgb_img.reshape((-1, 3))))
        points_3d = XYZRGB

        if labels is not None:
            labels_reshaped = labels.reshape((-1, n_label_channels))

        # Prune points
        if prune is True:
            valid = []
            for idx in range(points_3d.shape[0]):
                cur_y = points_3d[idx, 1]
                cur_z = points_3d[idx, 2]
                if (cur_z == 0) or (cur_z == 65535):  # Don't show things at 0 distance or max distance
                    continue
                elif (max_depth is not None) and (cur_z > max_depth):
                    continue
                elif (max_height is not None) and (cur_y > max_height):
                    continue
                elif (min_height is not None) and (cur_y < min_height):
                    continue
                else:
                    valid.append(idx)
            points_3d = points_3d[np.asarray(valid)]
            if labels is not None:
                labels_reshaped = labels_reshaped[np.asarray(valid)]

        if extrinsics is not None:
            points_3d = self.camera2world(extrinsics, points_3d)

        if labels is not None:
            points_3d_labels = np.hstack((points_3d[:, :3], labels_reshaped))
            return points_3d, points_3d_labels
        else:
            return points_3d

    @staticmethod
    def _camera2world_transform(no_rgb_points_3d, R, T):
        points_3d_world = (np.dot(R.T, no_rgb_points_3d.T) - T).T  # Nx3
        return points_3d_world

    @staticmethod
    def _world2camera_transform(no_rgb_points_3d, R, T):
        points_3d_world = (np.dot(R, no_rgb_points_3d.T + T)).T  # Nx3
        return points_3d_world

    def _transform_points(self, points_3d, extrinsics, transform):
        """Base/wrapper method for transforming points using R and T.
        """
        assert points_3d.ndim == 2
        orig_points_3d = points_3d
        points_3d = np.copy(orig_points_3d)
        if points_3d.shape[1] == 6:  # XYZRGB
            points_3d = points_3d[:, :3]
        elif points_3d.shape[1] == 3:  # XYZ
            points_3d = points_3d
        else:
            raise ValueError('3D points need to be XYZ or XYZRGB.')

        R, T = self.extrinsics2RT(extrinsics)
        points_3d_world = transform(points_3d, R, T)

        # Add color again (if appropriate)
        if orig_points_3d.shape[1] == 6:  # XYZRGB
            points_3d_world = np.hstack((points_3d_world, orig_points_3d[:, -3:]))
        return points_3d_world

    def camera2world(self, extrinsics, points_3d):
        """Transform from camera coordinates (3D) to world coordinates (3D).

        Args:
          points_3d: Nx3 or Nx6 matrix of N points with XYZ or XYZRGB values.
        """
        return self._transform_points(points_3d, extrinsics, self._camera2world_transform)

    def world2camera(self, extrinsics, points_3d):
        """Transform from world coordinates (3D) to camera coordinates (3D).
        """
        return self._transform_points(points_3d, extrinsics, self._world2camera_transform)

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @property
    def K_inv(self):
        return self._K_inv


def colorize_pointcloud(xyz, label, ignore_label=255):
    assert label[label != ignore_label].max() < len(COLOR_MAP_RGB), 'Not enough colors.'
    label_rgb = np.array([COLOR_MAP_RGB[i] if i != ignore_label else IGNORE_COLOR for i in label])
    return np.hstack((xyz, label_rgb))


class PlyWriter(object):
    POINTCLOUD_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                        ('blue', 'u1')]

    @classmethod
    def read_txt(cls, txtfile):
        # Read txt file and parse its content.
        with open(txtfile) as f:
            pointcloud = [l.split() for l in f]
        # Load point cloud to named numpy array.
        pointcloud = np.array(pointcloud).astype(np.float32)
        assert pointcloud.shape[1] == 6
        xyz = pointcloud[:, :3].astype(np.float32)
        rgb = pointcloud[:, 3:].astype(np.uint8)
        return xyz, rgb

    @staticmethod
    def write_ply(array, filepath):
        ply_el = PlyElement.describe(array, 'vertex')
        target_path, _ = os.path.split(filepath)
        if target_path != '' and not os.path.exists(target_path):
            os.makedirs(target_path)
        PlyData([ply_el]).write(filepath)

    @classmethod
    def write_vertex_only_ply(cls, vertices, filepath):
        # assume that points are N x 3 np array for vertex locations
        color = 255 * np.ones((len(vertices), 3))
        pc_points = np.array([tuple(p) for p in np.concatenate((vertices, color), axis=1)],
                             dtype=cls.POINTCLOUD_DTYPE)
        cls.write_ply(pc_points, filepath)

    @classmethod
    def write_ply_vert_color(cls, vertices, colors, filepath):
        # assume that points are N x 3 np array for vertex locations
        pc_points = np.array([tuple(p) for p in np.concatenate((vertices, colors), axis=1)],
                             dtype=cls.POINTCLOUD_DTYPE)
        cls.write_ply(pc_points, filepath)

    @classmethod
    def concat_label(cls, target, xyz, label):
        subpointcloud = np.concatenate([xyz, label], axis=1)
        subpointcloud = np.array([tuple(l) for l in subpointcloud], dtype=cls.POINTCLOUD_DTYPE)
        return np.concatenate([target, subpointcloud], axis=0)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Point cloud IO
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

# Mesh IO
import trimesh
import math

import matplotlib.pyplot as pyplot


# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b, :, :], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)


def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize, vsize, vsize, num_sample, 3))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i, j, k) not in loc2pc:
                    vol[i, j, k, :, :] = np.zeros((num_sample, 3))
                else:
                    pc = loc2pc[(i, j, k)]  # a list of (3,) arrays
                    pc = np.vstack(pc)  # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0] > num_sample:
                        pc = random_sampling(pc, num_sample, False)
                    elif pc.shape[0] < num_sample:
                        pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i, j, k]) + 0.5) * voxel - radius
                    pc = (pc - pc_center) / voxel  # shift and scale
                    vol[i, j, k, :, :] = pc
    return vol


def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b, :, :], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2 * radius / float(imgsize)
    locations = (points[:, 0:2] + radius) / pixel  # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i, j) not in loc2pc:
                img[i, j, :, :] = np.zeros((num_sample, 3))
            else:
                pc = loc2pc[(i, j)]
                pc = np.vstack(pc)
                if pc.shape[0] > num_sample:
                    pc = random_sampling(pc, num_sample, False)
                elif pc.shape[0] < num_sample:
                    pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                pc_center = (np.array([i, j]) + 0.5) * pixel - radius
                pc[:, 0:2] = (pc[:, 0:2] - pc_center) / pixel
                img[i, j, :, :] = pc
    return img


# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i, :]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix


def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


# ----------------------------------------
# BBox
# ----------------------------------------
def bbox_corner_dist_measure(crnr1, crnr2):
    """ compute distance between box corners to replace iou
    Args:
        crnr1, crnr2: Nx3 points of box corners in camera axis (y points down)
        output is a scalar between 0 and 1
    """

    dist = sys.maxsize
    for y in range(4):
        rows = ([(x + y) % 4 for x in range(4)] + [4 + (x + y) % 4 for x in range(4)])
        d_ = np.linalg.norm(crnr2[rows, :] - crnr1, axis=1).sum() / 8.0
        if d_ < dist:
            dist = d_

    u = sum([np.linalg.norm(x[0, :] - x[6, :]) for x in [crnr1, crnr2]]) / 2.0

    measure = max(1.0 - dist / u, 0)
    print(measure)

    return measure


def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths
    """
    which_dim = len(points.shape) - 2  # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5 * (mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)


def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """

    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    mesh_list.export(out_filename)

    return


def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    mesh_list.export(out_filename)

    return


def generate_bbox_mesh(bbox, output_file=None):
    """
    bbox: np array (n, 6),
    output_file: string
    """

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                indices.append(
                    np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.02
    offset = [0, 0, 0]
    verts = []
    indices = []
    for box in bbox:
        box_min = np.array([box[0], box[1], box[2]])
        box_max = np.array([box[3], box[4], box[5]])
        edges = get_bbox_edges(box_min, box_max)
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)

    return verts, indices


def write_oriented_bbox_(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def write_ply_mesh(verts, colors, indices, output_file):
        if colors is None:
            colors = np.zeros_like(verts)
        if indices is None:
            indices = []

        file = open(output_file, 'w')
        file.write('ply \n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {:d}\n'.format(len(verts)))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('element face {:d}\n'.format(len(indices)))
        file.write('property list uchar uint vertex_indices\n')
        file.write('end_header\n')
        for vert, color in zip(verts, colors):
            file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0] * 255),
                                                                int(color[1] * 255), int(color[2] * 255)))
        for ind in indices:
            file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
        file.close()

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box = np.array([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]])
        box[:, 0] = box[:, 0] * lengths[0] + trns[0, 3]
        box[:, 1] = box[:, 1] * lengths[1] + trns[1, 3]
        box[:, 2] = box[:, 2] * lengths[2] + trns[2, 3]
        box[:, 3] = box[:, 3] * lengths[0] + trns[0, 3]
        box[:, 4] = box[:, 4] * lengths[1] + trns[1, 3]
        box[:, 5] = box[:, 5] * lengths[2] + trns[2, 3]
        vertices, indices = generate_bbox_mesh(box)
        return vertices, indices

    verts, inds = convert_oriented_box_to_trimesh_fmt(scene_bbox)
    write_ply_mesh(verts, None, inds, out_filename)

    return


def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[1, 1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0, :] = np.array([cosval, 0, sinval])
        rotmat[2, :] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    mesh_list.export(out_filename)

    return


def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src, tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0, 0, 1], vec, False)
        vec = tgt - src  # compute again since align_vectors modifies vec in-place!
        M[:3, 3] = 0.5 * src + 0.5 * tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(trimesh.creation.cylinder(radius=rad, height=height, sections=res, transform=M))
    mesh_list = trimesh.util.concatenate(scene.dump())
    mesh_list.export('%s.ply' % (filename))


# ----------------------------------------
# Testing
# ----------------------------------------
if __name__ == '__main__':
    print('running some tests')

    ############
    ## Test "write_lines_as_cylinders"
    ############
    pcl = np.random.rand(32, 2, 3)
    write_lines_as_cylinders(pcl, 'point_connectors')
    input()

    scene_bbox = np.zeros((1, 7))
    scene_bbox[0, 3:6] = np.array([1, 2, 3])  # dx,dy,dz
    scene_bbox[0, 6] = np.pi / 4  # 45 degrees
    write_oriented_bbox(scene_bbox, 'single_obb_45degree.ply')
    ############
    ## Test point_cloud_to_bbox
    ############
    pcl = np.random.rand(32, 16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (32, 6)

    pcl = np.random.rand(16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (6,)

    ############
    ## Test corner distance
    ############
    crnr1 = np.array([[2.59038660e+00, 8.96107932e-01, 4.73305349e+00],
                      [4.12281644e-01, 8.96107932e-01, 4.48046631e+00],
                      [2.97129656e-01, 8.96107932e-01, 5.47344275e+00],
                      [2.47523462e+00, 8.96107932e-01, 5.72602993e+00],
                      [2.59038660e+00, 4.41155793e-03, 4.73305349e+00],
                      [4.12281644e-01, 4.41155793e-03, 4.48046631e+00],
                      [2.97129656e-01, 4.41155793e-03, 5.47344275e+00],
                      [2.47523462e+00, 4.41155793e-03, 5.72602993e+00]])
    crnr2 = crnr1

    print(bbox_corner_dist_measure(crnr1, crnr2))

    print('tests PASSED')

