import copy
import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch

import numba


@numba.jit
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def update_idx2hasPoint_np(idx2hasPoint, new_indexes, new_coords, tag):
    if new_indexes.dtype == np.bool:  # to compatible with [True, False] style retrieval
        new_indexes = np.nonzero(np.array(new_indexes, dtype=np.int))[0]

    # logging.info("===> tag: {}, new_indexes[:10]: {}".format(tag, new_indexes[:10]))

    new_indexes = np.array(new_indexes, dtype=np.int)

    # new_indexes required to be fully sorted
    if not is_sorted(new_indexes):
        raise ValueError(
            "The new_indexes required to be fully sorted. Error occur for tag: {}, new_indexes[:10]: {}".format(
                tag, new_indexes[:10]))

    mask_idx2hasPoint = np.nonzero(idx2hasPoint)[0]
    mask_idx2hasPoint = mask_idx2hasPoint[new_indexes]
    # assert there is point
    assert np.unique(idx2hasPoint[mask_idx2hasPoint]) == [1]
    idx2hasPoint.fill(0)
    idx2hasPoint[mask_idx2hasPoint] = 1

    assert np.sum(idx2hasPoint) == new_coords.shape[0], 'np.sum(idx2hasPoint): {}, new_coords.shape[0]: {}'.format(
        np.sum(idx2hasPoint), new_coords.shape[0]
    )
    assert np.sum(idx2hasPoint) == new_indexes.shape[0], 'np.sum(idx2hasPoint): {}, new_mapping.shape[0]; {}'.format(
        np.sum(idx2hasPoint), new_indexes.shape[0]
    )

    return idx2hasPoint


def cal_num_drop_label(labels_before, labels_after, ignore_label, tag):
    num_before = np.sum(np.array(labels_before != ignore_label, dtype=np.int))
    num_after = np.sum(np.array(labels_after != ignore_label, dtype=np.int))
    # filtered by tag
    # ['PREVOXELIZATION_VOXEL_SIZE', 'clip_inds', 'RandomDropout', 'ME.utils.sparse_quantize']

    """
    for scannet
    (1) PREVOXELIZATION_VOXEL_SIZE does not lose points
    (2) clip_inds does not lose points
    (3) RandomDropout lose about 10%-40% labels
    (4) ME.utils.sparse_quantize lose about 10%-40% labels
    """

    """
    for stanford,
    (1) PREVOXELIZATION_VOXEL_SIZE does not lose points
    (2) clip_inds lose about 30%-50% labels at sometimes
    (3) RandomDropout lose about 10%-25% labels
    (4) ME.utils.sparse_quantize lose about 50%-80% labels
    """

    # if tag != 'PREVOXELIZATION_VOXEL_SIZE':
    # if tag != 'clip_inds':
    # if tag != 'RandomDropout':
    # if tag != 'ME.utils.sparse_quantize':
    #     return
    logging.info("===>Tag: {}, num_before: {}, num_after: {}, drop percentage: {}%, drop_or_not: {}".format(
        tag, num_before, num_after, round((1 - (num_after / num_before)) * 100., 2), num_before != num_after
    ))


def merge_mapping_index(mapping, label_mask):
    label_indexes = np.nonzero(np.array(label_mask, dtype=np.int))
    # np.union1d will sort the indexes
    # Return the unique, sorted array of values that are in either of the two input arrays.
    return np.union1d(mapping, label_indexes)


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels, idx2hasPoint):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'trans_range_ratio={0})'.format(self.trans_range_ratio)
        return format_string


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels, idx2hasPoint):
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            scale = 255 / (hi - lo)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'randomize_blend_factor={0}'.format(self.randomize_blend_factor)
        format_string += ', blend_factor={0})'.format(self.blend_factor)
        return format_string


class ChromaticJitter(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels, idx2hasPoint):
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'std={0})'.format(self.std)
        return format_string


class HueSaturationTranslation(object):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels, idx2hasPoint):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'hue_max={0}'.format(self.hue_max)
        format_string += ', saturation_max={0})'.format(self.saturation_max)
        return format_string


def reversed_index_1d(num_elem, index_to_reverse):
    assert index_to_reverse.dim() == 1, 'index_to_reverse.dim(): {}'.format(index_to_reverse.dim())
    return torch.tensor([_ for _ in range(num_elem) if _ not in index_to_reverse])


##############################
# Coordinate transformations
##############################
class RandomDropout(object):

    def __init__(self,
                 dropout_ratio=0.2, dropout_application_ratio=0.5,
                 retain_label=False, ignore_label=255):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        super(RandomDropout, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio
        self.retain_label = retain_label
        self.ignore_label = ignore_label

    def __call__(self, coords, feats, labels, idx2hasPoint):
        if random.random() < self.dropout_ratio:
            N = len(coords)
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
            if not self.retain_label:
                inds = np.sort(inds)
            else:
                inds = merge_mapping_index(inds, label_mask=labels != self.ignore_label)

            coords, feats, labels = coords[inds], feats[inds], labels[inds]

            idx2hasPoint = update_idx2hasPoint_np(idx2hasPoint, inds, coords, tag='RandomDropout')

        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'dropout_ratio={0}'.format(self.dropout_ratio)
        format_string += ', dropout_application_ratio={0})'.format(self.dropout_application_ratio)
        return format_string


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels, idx2hasPoint):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'upright_axis={0}'.format(self.upright_axis)
        format_string += ', is_temporal={0})'.format(self.is_temporal)
        return format_string


class ElasticDistortion:

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords, feats, labels

    def __call__(self, coords, feats, labels, idx2hasPoint):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity,
                                                                    magnitude)
        return coords, feats, labels, idx2hasPoint

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'distortion_params={0})'.format(self.distortion_params)
        return format_string


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class cfl_collate_fn_factory:
    """Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints, two_stream=False, alignment_level='feature', ignore_label=None):
        self.limit_numpoints = limit_numpoints
        self.two_stream = two_stream
        assert alignment_level in ['feature', 'input'], 'Unsupported alignment level ``{}'''.format(alignment_level)
        self.alignment_level = alignment_level
        self.ignore_label = ignore_label
        if self.two_stream:
            if self.ignore_label is None:
                raise ValueError("ignore_label must be provided for two_stream dataloader")

    def __call__(self, batch_data):
        if not self.two_stream:
            return self._call(batch_data)
        else:
            return self._two_stream_call(batch_data)

    def _call(self, batch_data):
        coords = self._aggregate(batch_data, 'coords')
        feats = self._aggregate(batch_data, 'feats')
        labels = self._aggregate(batch_data, 'labels')
        indexes = self._aggregate(batch_data, 'indexes')
        # inverse_map = self._aggregate(batch_data, 'inverse_map')
        # instance_labels = self._aggregate(batch_data, 'instance_labels')

        coords_batch, feats_batch, labels_batch, = [], [], [],
        indexes_batch, inverse_map_batch, instance_labels_batch = [], [], []

        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}. '
                )
                break
            # coords_batch.append(
            #     torch.cat((torch.from_numpy(
            #        coords[batch_id]).int(), torch.ones(num_points, 1).int() * batch_id), 1))
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1)
            )
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).long())
            indexes_batch.append(torch.tensor(indexes[batch_id]).long())
            # inverse_map_batch.append(torch.tensor(inverse_map[batch_id]).long())
            # instance_labels_batch.append(torch.tensor(instance_labels[batch_id]).long())

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).long()
        indexes_batch = torch.stack(indexes_batch, 0).long()
        # inverse_map_batch = torch.cat(inverse_map_batch, 0).long()
        # instance_labels_batch = torch.cat(instance_labels_batch, 0).long()
        return {
            'coords': coords_batch, 'feats': feats_batch, 'labels': labels_batch, 'indexes': indexes_batch,
            # 'inverse_map': inverse_map_batch,
            # 'instance_labels': instance_labels_batch
        }

    def _two_stream_call(self, batch_data):
        coords = self._aggregate(batch_data, 'coords')
        feats = self._aggregate(batch_data, 'feats')
        labels = self._aggregate(batch_data, 'labels')

        coords_aux = self._aggregate(batch_data, 'coords_aux')
        feats_aux = self._aggregate(batch_data, 'feats_aux')
        labels_aux = self._aggregate(batch_data, 'labels_aux')

        idx2hasPoints = self._aggregate(batch_data, 'idx2hasPoints')
        idx2hasPoints_aux = self._aggregate(batch_data, 'idx2hasPoints_aux')

        coords_batch, feats_batch, labels_batch, corr_indexes_batch = [], [], [], []
        coords_aux_batch, feats_aux_batch, labels_aux_batch, corr_indexes_aux_batch = [], [], [], []

        # num_label_ori = 0
        # num_label_align = 0

        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            idx2hasPoint = idx2hasPoints[batch_id]
            idx2hasPoint_aux = idx2hasPoints_aux[batch_id]
            # logging.info("===> coord.size(): {}, coord_aux.size(): {}".format(
            #     coords[batch_id].shape, coords_aux[batch_id].shape
            # ))
            corr_index, corr_index_aux = self.get_corresponding_index(idx2hasPoint, idx2hasPoint_aux)

            # get all cur batch data here
            coord = coords[batch_id]
            coord_aux = coords_aux[batch_id]

            feat = feats[batch_id]
            feat_aux = feats_aux[batch_id]

            label = labels[batch_id]
            label_aux = labels_aux[batch_id]

            self.check_corresponding_label(label[corr_index], label_aux[corr_index_aux])

            if self.alignment_level == 'input':
                # align here
                coord = coords[batch_id][corr_index]
                coord_aux = coords_aux[batch_id][corr_index_aux]

                feat = feats[batch_id][corr_index]
                feat_aux = feats_aux[batch_id][corr_index_aux]

                label = labels[batch_id][corr_index]
                label_aux = labels_aux[batch_id][corr_index_aux]

                # update the corr_index to be compatible with the newest API
                corr_index = np.arange(coord.shape[0])
                corr_index_aux = np.arange(coord_aux.shape[0])

                num_p, num_p_aux = coord.shape[0], coord_aux.shape[0]
                num_l, num_l_aux = np.sum(label != 255), np.sum(label_aux != 255)
                # valid or not, there may be no point or no label after alignment. Only find this issue on S3DIS
                # we do a little hack here
                # we simply copy one stream data to another stream
                if (num_p < 100 or num_p_aux < 100) or (num_l < 1 or num_l_aux < 1):
                    logging.warning('Invalid corresponding data is found, hacked copying is applied')
                    coord = coords[batch_id]
                    coord_aux = copy.deepcopy(coord)

                    feat = feats[batch_id]
                    feat_aux = copy.deepcopy(feat) + 0.01 * np.random.randn(
                        feat.shape[0], feat.shape[1])  # apply a random var

                    label = labels[batch_id]
                    label_aux = copy.deepcopy(label)

                    # update the corr_index to be compatible with the newest API
                    corr_index = np.arange(coord.shape[0])
                    corr_index_aux = np.arange(coord_aux.shape[0])

            # update num_points at last
            num_points = coord.shape[0]
            num_points_aux = coord_aux.shape[0]

            batch_num_points += (num_points + num_points_aux)
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords) + sum(len(c) for c in coords_aux)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points - num_points_aux}. '
                )
                break

            coords_batch.append(torch.cat([torch.ones(num_points, 1).int() * batch_id,
                                           torch.from_numpy(coord).int()], 1))
            feats_batch.append(torch.from_numpy(feat))
            labels_batch.append(torch.from_numpy(label).long())
            corr_indexes_batch.append(torch.cat([torch.ones(corr_index.shape[0], 1).long() * batch_id,
                                                 torch.from_numpy(corr_index).unsqueeze(1).long()], 1))

            coords_aux_batch.append(torch.cat([torch.ones(num_points_aux, 1).int() * batch_id,
                                               torch.from_numpy(coord_aux).int()], 1))
            feats_aux_batch.append(torch.from_numpy(feat_aux))
            labels_aux_batch.append(torch.from_numpy(label_aux).long())
            corr_indexes_aux_batch.append(torch.cat([torch.ones(corr_index_aux.shape[0], 1).long() * batch_id,
                                                     torch.from_numpy(corr_index_aux).unsqueeze(1).long()], 1))

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).long()
        corr_indexes_batch = torch.cat(corr_indexes_batch, 0).long()

        coords_aux_batch = torch.cat(coords_aux_batch, 0).int()
        feats_aux_batch = torch.cat(feats_aux_batch, 0).float()
        labels_aux_batch = torch.cat(labels_aux_batch, 0).long()
        corr_indexes_aux_batch = torch.cat(corr_indexes_aux_batch, 0)

        # logging.info("===> coords_batch.size(): {}, coords_aux_batch.size(): {}".format(
        #     coords_batch.size(), coords_aux_batch.size()
        # ))

        # assert coords_batch.size() == coords_aux_batch.size()

        return {
            'coords': coords_batch, 'feats': feats_batch, 'labels': labels_batch,
            'coords_aux': coords_aux_batch, 'feats_aux': feats_aux_batch, 'labels_aux': labels_aux_batch,
            'corr_indexes_batch': corr_indexes_batch, 'corr_indexes_aux_batch': corr_indexes_aux_batch,
        }

    @staticmethod
    def _aggregate(batch_data, key):
        return [_[key] for _ in batch_data]

    @staticmethod
    def get_corresponding_index(idx2hasPoint1, idx2hasPoint2):
        # idx2hasPoint1 = idx2hasPoint1.long()
        # idx2hasPoint2 = idx2hasPoint2.long()

        num_points1 = np.sum(idx2hasPoint1)
        num_points2 = np.sum(idx2hasPoint2)

        # logging.info("===> num_points1: {}, num_points2: {}".format(
        #     num_points1, num_points2
        # ))
        #
        # assert num_points1 == num_points2, 'num_points1={} does not equal to num_points2={}'.format(
        #     num_points1, num_points2
        # )

        p = np.array(idx2hasPoint1 * idx2hasPoint2, dtype=np.bool)

        corr_index1 = np.zeros_like(idx2hasPoint1) * (idx2hasPoint1.shape[0] + 1)  # will trigger error if not right
        corr_index1[np.array(idx2hasPoint1, dtype=np.bool)] = np.arange(num_points1)
        corr_index1 = corr_index1[p]

        corr_index2 = np.zeros_like(idx2hasPoint2) * (idx2hasPoint2.shape[0] + 1)  # will trigger error if not right
        corr_index2[np.array(idx2hasPoint2, dtype=np.bool)] = np.arange(num_points2)
        corr_index2 = corr_index2[p]

        corr_index1, corr_index2 = np.array(corr_index1, dtype=np.int), np.array(corr_index2, dtype=np.int)

        # logging.info("===> corr_index1: {}, corr_index1.shape: {}".format(corr_index1, corr_index1.shape))
        # logging.info("===> corr_index2: {}, corr_index2.shape: {}".format(corr_index2, corr_index2.shape))

        assert corr_index1.shape == corr_index2.shape  # shape may be (0,), meaning no corresponding part
        # assert corr_index1.shape[0] > 0, 'corr_index1.shape: {}'.format(corr_index1.shape)

        return corr_index1, corr_index2

    def check_corresponding_label(self, label1, label2):
        # filter out ignore label first
        # ignore_label will be generated during label voxelization (e.g., sparse_label = True by default)

        #   1, 255, 255,  4, 3, 255
        # 255, 255,    4, 4, 3,   1

        retain_mask1 = label1 != self.ignore_label
        retain_mask2 = label2 != self.ignore_label

        retain_mask = np.logical_and(retain_mask1, retain_mask2)
        l1_retained, l2_retained = label1[retain_mask], label2[retain_mask]
        if l1_retained.shape[0] == 0:
            return
        equal_flag = np.mean(np.array(label1[retain_mask] == label2[retain_mask], dtype=np.float))
        assert equal_flag >= 1.0, 'equal_flag={} >= 1.0 does not hold, l1={}, l2={}'.format(
            equal_flag, l1_retained, l2_retained)


class cflt_collate_fn_factory(cfl_collate_fn_factory):
    """Generates collate function for coords, feats, labels, point_clouds, transformations.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints, two_stream=False, alignment_level='feature', ignore_label=None):
        super(cflt_collate_fn_factory, self).__init__(limit_numpoints, two_stream=two_stream,
                                                      alignment_level=alignment_level, ignore_label=ignore_label)

    def __call__(self, batch_data):
        if not self.two_stream:
            return self._t_call(batch_data)
        else:
            return self._t_two_stream_call(batch_data)

    def _t_call(self, batch_data):
        batch_ret = super(cflt_collate_fn_factory, self).__call__(batch_data)
        transformations = self._aggregate(batch_data, 'transformations')

        coords_batch = batch_ret['coords']
        num_truncated_batch = coords_batch[:, -1].max().item() + 1

        batch_id = 0
        transformations_batch = []
        for transformation in transformations:
            if batch_id >= num_truncated_batch:
                break
            transformations_batch.append(torch.from_numpy(transformation).float())
            batch_id += 1
        transformations_batch = torch.stack(transformations_batch, 0)

        batch_ret['transformations'] = transformations_batch

        return batch_ret

    def _t_two_stream_call(self, batch_data):
        batch_ret = super().__call__(batch_data)
        transformations = self._aggregate(batch_data, 'transformations')
        transformations_aux = self._aggregate(batch_data, 'transformations_aux')

        coords_batch = batch_ret['coords']
        num_truncated_batch = coords_batch[:, -1].max().item() + 1

        batch_id = 0
        transformations_batch = []
        transformations_aux_batch = []
        for transformation, transformation_aux in zip(transformations, transformations_aux):
            if batch_id >= num_truncated_batch:
                break
            transformations_batch.append(torch.from_numpy(transformation).float())
            transformations_aux_batch.append(torch.from_numpy(transformation_aux).float())
            batch_id += 1
        transformations_batch = torch.stack(transformations_batch, 0)
        transformations_aux_batch = torch.stack(transformations_aux_batch, 0)

        batch_ret['transformations'] = transformations_batch
        batch_ret['transformations_aux'] = transformations_aux_batch

        return batch_ret
