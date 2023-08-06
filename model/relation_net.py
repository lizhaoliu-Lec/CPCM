"""
Reference: https://github.com/liuzhengzhe/One-Thing-One-Click/blob/master/relation/model/pointgroup/pointgroup.py
"""

import functools
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from spconv import SparseConvTensor
from spconv import SparseSequential
from spconv import SparseConv3d, SparseInverseConv3d, SubMConv3d

from .build import MODEL_REGISTRY

from model.common import ResidualBlock, VGGBlock


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(inplace=True),
                SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False,
                             indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            if nPlanes[0] == 32:
                self.deconv = SparseSequential(
                    norm_fn(nPlanes[1]),
                    nn.ReLU(inplace=True),
                    SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False,
                                        indice_key='spconv{}'.format(indice_key_id))
                )
            else:
                self.deconv = SparseSequential(
                    norm_fn(nPlanes[1]),
                    nn.ReLU(inplace=True),
                    SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False,
                                        indice_key='spconv{}'.format(indice_key_id))
                )

            blocks_tail = {}

            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn,
                                                         indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = SparseSequential(blocks_tail)

    def forward(self, _input):
        output = self.blocks(_input)
        identity = SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


@MODEL_REGISTRY.register()
class RelationNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

        input_c = cfg.DATA.input_channel
        base_dim = cfg.MODEL.base_dim
        num_classes = cfg.DATA.num_classes
        block_reps = cfg.MODEL.block_reps
        block_residual = cfg.MODEL.block_residual
        group = cfg.MODEL.group

        self.base_dim = base_dim
        self.num_classes = num_classes
        self.cluster_radius = group.cluster_radius
        self.cluster_meanActive = group.cluster_meanActive
        self.cluster_shift_meanActive = group.cluster_shift_meanActive
        self.cluster_npoint_thre = group.cluster_npoint_thre

        self.score_scale = cfg.MODEL.score_scale
        self.score_fullscale = cfg.MODEL.score_fullscale
        self.mode = cfg.MODEL.score_mode

        self.num_sample_per_class = cfg.MODEL.num_sample_per_class
        self.temperature = cfg.MODEL.temperature
        self.momentum = cfg.MODEL.momentum
        self.ignore_label = cfg.DATA.ignore_label

        # self.pretrain_path = cfg.TRAIN.pretrain_path
        # self.pretrain_module = cfg.TRAIN.pretrain_module
        # self.fix_module = cfg.TRAIN.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.MODEL.use_coords:
            input_c += 3

        # backbone
        self.input_conv = SparseSequential(
            SubMConv3d(input_c, base_dim, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock(
            [base_dim, 2 * base_dim, 3 * base_dim, 4 * base_dim, 5 * base_dim, 6 * base_dim, 7 * base_dim], norm_fn,
            block_reps, block, indice_key_id=1)

        self.output_layer = SparseSequential(
            norm_fn(base_dim)
        )

        # TODO ori code for is 10 instead of self.num_classes
        self.prototypes = torch.zeros(self.num_classes, self.base_dim, requires_grad=False)
        self.start = True

    def loss_fn(self, scores, labels):
        # scores: (N, nClass), float32, cuda
        # labels: (N), long, cuda
        return nn.CrossEntropyLoss(ignore_index=self.config.DATA.ignore_label)(scores, labels)

    @staticmethod
    def set_bn_init(base_dim):
        classname = base_dim.__class__.__name__
        if classname.find('BatchNorm') != -1:
            base_dim.weight.data.fill_(1.0)
            base_dim.bias.data.fill_(0.0)

    @staticmethod
    def clusters_voxelization(clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        from libirary.pointgroup_ops.functions import pointgroup_ops

        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :param fullscale:
        :param scale:
        :param mode:
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0,
                                                  clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[
            0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        _range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - _range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(
            fullscale - _range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()],
                                    1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1,
                                                                       mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape,
                                              int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def forward(self, _input, input_map, coords, batch_idxs, batch_offsets, group2points, groups=None, labels=None):
        """
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        :param group2points: 
        :param groups: 
        :param labels: (N), int, cuda, optional, only required during training
        """
        ret = {}

        output = self.input_conv(_input)
        # print ('0',output.features)
        output = self.unet(output)
        # print ('1',output.features)
        output = self.output_layer(output)
        # print ('2',output.features)
        output_feats = output.features[input_map.long()]

        ret['semantic_feats'] = output_feats

        # produce the semantic prediction here
        with torch.no_grad():
            semantic_scores = torch.matmul(output_feats, self.prototypes.to(
                output_feats.device).T)  # (bs, dim) x (dim, nc) => (bs, nc)
            ret['semantic_scores'] = semantic_scores

        if self.training:
            if labels is None:
                raise ValueError('labels are required during training')
            if groups is None:
                raise ValueError('groups are required during training')

            nc = self.num_classes
            ns = self.num_sample_per_class
            tmp_feats = torch.zeros(nc, ns, self.base_dim).to(output_feats.device)
            tmp_labels = torch.zeros(nc * ns).fill_(self.ignore_label).long().to(output_feats.device)

            clsId_batchId_segIds = [[] for _ in range(nc)]
            ignore_classes = []

            # groups mean for a batch of group
            # group mean for: label id to point id

            # (1) gather points for each class, while maintaining their corresponding batch_id
            for batch_id, group in enumerate(groups):
                for class_id in range(nc):
                    for seg_id in group[class_id]:
                        # TODO the class_id here is redundant
                        clsId_batchId_segIds[class_id].append((class_id, batch_id, seg_id))

            # (2) sample ns points for each class
            for class_id in range(nc):
                class_points = clsId_batchId_segIds[class_id]
                num_class_segments = len(class_points)
                # shuffle points for this class, in order to achieve randomly sampling
                random.shuffle(class_points)

                if num_class_segments == 0:  # if no points for this class, we ignore this class
                    ignore_classes.append(class_id)
                elif num_class_segments >= ns:  # if points are sufficient, we retrieve the first ns points
                    class_points = class_points[:ns]
                else:
                    # replace = True means for repeatedly sampling [1, 2, 3] => [1, 1, 2, 3, 3]
                    repeated_sampling_indexes = np.random.choice(np.arange(num_class_segments), size=ns, replace=True)
                    class_points = [class_points[_] for _ in repeated_sampling_indexes.tolist()]

                # update the class points after sampling
                clsId_batchId_segIds[class_id] = class_points

            batch_sampled_features = []  # TODO, a more elegant way to collect tmp features and labels
            batch_sampled_labels = []
            # (3) update feature and label for each sampled point
            for class_id in range(nc):
                # filter out the ignored class
                if class_id in ignore_classes:
                    continue

                for sampled_id in range(ns):
                    class_id, batch_id, seg_id = clsId_batchId_segIds[class_id][sampled_id]
                    point_idxes = torch.tensor(np.array(group2points[batch_id][seg_id])) + batch_offsets[batch_id]
                    sampled_features = output_feats[point_idxes.to(output_feats.device)]  # (1, dim)
                    tmp_feats[class_id, sampled_id, :] = torch.mean(sampled_features, dim=0)
                    tmp_labels[class_id * ns + sampled_id] = class_id

            # (4) calculate loss with all features and class prototypes
            product = torch.matmul(output_feats,  # (nc * ns, dim)
                                   self.prototypes.to(
                                       output_feats.device).T) / self.temperature  # (nc * ns, dim) x (dim, nc) => (nc * ns, nc)

            loss = self.loss_fn(product, labels)
            ret['loss'] = loss

            # (5) update class prototypes by momentum average
            tmp_feats_data = torch.mean(tmp_feats, dim=1).to(self.prototypes.device).data  # (nc, dim)
            # TODO we should not update prototypes w.r.t ignore classes
            if self.start:
                self.start = False
                self.prototypes.data = tmp_feats_data
            else:
                m = self.momentum
                concerned_classes = [_ for _ in range(nc) if _ not in ignore_classes]
                # print("===> concerned_classes： {}".format(concerned_classes))
                concerned_class_indexes = torch.tensor(concerned_classes, dtype=torch.long,
                                                       device=self.prototypes.device)
                self.prototypes.data[concerned_class_indexes, :] = m * self.prototypes.data[concerned_class_indexes,
                                                                       :] + (1 - m) * tmp_feats_data[
                                                                                      concerned_class_indexes, :]

        return ret
