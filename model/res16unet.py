# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import warnings

import torch
import torch.nn as nn
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiReLU

from model.common.transformer_block import TransformerBlock, MultiHeadAttention, TransformerBlockOri
from model.resnet import ResNetBase, get_norm
from model.common.base import ConvType, NormType, conv, conv_tr
from model.common.resnet_block import BasicBlock, Bottleneck

import MinkowskiEngine.MinkowskiOps as MO

from util.utils import euclidean_distance
from .build import MODEL_REGISTRY


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Res16UNetBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, config):
        self.config = config
        if not hasattr(self.config.MODEL, 'D'):
            self.config.MODEL.D = 3
        super(Res16UNetBase, self).__init__(config)
        self.mlp_check_pseudo_instance_label = config.MODEL.mlp_check_pseudo_instance_label
        self.pic_branch = config.MODEL.pic_branch
        self.pic_attentive_pool = config.MODEL.pic_attentive_pool

    def is_valid_pseudo_instance_label(self, labels):
        is_valid = True

        # filter out the invalid labels
        filtered_labels = labels.clone()
        label_unique, label_count = torch.unique(filtered_labels[filtered_labels != self.config.DATA.ignore_label],
                                                 return_counts=True)

        # three case
        # (0) if no label available
        if label_unique.size(0) == 0:
            is_valid = False
        # (1) if all the same, no_neg
        if label_unique.size(0) == 1:
            is_valid = False
        # (2) if all unique, no_pos
        if label_unique.size(0) == label_count.sum():
            is_valid = False

        if not is_valid:
            logging.warning("Invalid pseudo instance label found, label_unique={}".format(label_unique))

        return is_valid

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.MODEL.bn_momentum
        mlp = config.MODEL.mlp
        mlp_factor = config.MODEL.mlp_factor
        pic_branch = config.MODEL.pic_branch
        pic_d_model = config.MODEL.pic_dmodel
        pic_heads = config.MODEL.pic_heads
        pic_dff = config.MODEL.pic_dff
        pic_dropout = config.MODEL.pic_dropout

        pic_attentive_pool = config.MODEL.pic_attentive_pool
        pic_attentive_pool_with_coord = config.MODEL.pic_attentive_pool_with_coord
        pic_attentive_pool_num_head = config.MODEL.pic_attentive_pool_num_head
        pic_attentive_pool_dropout = config.MODEL.pic_attentive_pool_dropout

        # apply scene cls
        apply_scene_cls = config.MODEL.apply_scene_cls

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.MODEL.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
        self.relu = MinkowskiReLU(inplace=True)

        self.mlp = None
        if mlp:
            self.mlp = self.make_mlp_layer(self.PLANES[7], mlp_factor)

        if pic_branch:
            # self.pic_block5 = TransformerBlock(d_model_in=self.PLANES[4],
            #                                    d_model=pic_d_model,
            #                                    heads=pic_heads,
            #                                    d_ff=pic_dff,
            #                                    dropout=pic_dropout)
            # self.pic_block6 = TransformerBlock(d_model_in=self.PLANES[5],
            #                                    d_model=pic_d_model,
            #                                    heads=pic_heads,
            #                                    d_ff=pic_dff,
            #                                    dropout=pic_dropout)
            # self.pic_block7 = TransformerBlock(d_model_in=self.PLANES[6],
            #                                    d_model=pic_d_model,
            #                                    heads=pic_heads,
            #                                    d_ff=pic_dff,
            #                                    dropout=pic_dropout)
            self.pic_block8 = TransformerBlock(d_model_in=self.PLANES[7],
                                               d_model=pic_d_model,
                                               heads=pic_heads,
                                               d_ff=pic_dff,
                                               dropout=pic_dropout)

        if pic_attentive_pool:
            d_model = self.PLANES[7]

            if pic_attentive_pool_with_coord:
                d_model += 3  # plus 3 coordinates features

            # self.pic_embeddings = nn.Parameter(torch.zeros(config.MODEL.out_channels, d_model))
            # trunc_normal_(self.pic_embeddings, std=.02)

            self.pic_attentive_pool_module = TransformerBlockOri(d_model=d_model, d_ff=d_model,
                                                                 heads=pic_attentive_pool_num_head,
                                                                 dropout=pic_attentive_pool_dropout)

        if apply_scene_cls:
            self.final_scene = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)

    def make_mlp_layer(self, in_channels, inner_factor):
        mid_channels = int(in_channels * inner_factor)
        return nn.Sequential(*[
            conv(in_channels, mid_channels, kernel_size=1, stride=1, bias=True, D=self.D),
            MinkowskiReLU(inplace=True),
            conv(mid_channels, in_channels, kernel_size=1, stride=1, bias=True, D=self.D),
        ])

    def forward(self, x, labels=None):
        ret = {}

        if self.training and self.pic_branch:
            if labels is None:
                raise ValueError('labels are required during training')
            out, pic_out, pic_labels = self._pic_forward(x, labels)
            ret['pic_feats'] = pic_out
            ret['pic_labels'] = pic_labels
            # TODO plus standard cross entropy loss for pic feats and labels
            # which not only solve optimize all params problem
            # but also may bring improvements
        else:
            out = self._base_forward(x)
        # ori output
        # output0: self.final(out)
        # output1: out

        sparse_scores = self.final(out)

        # logging.info("===> sparse_scores: \n{}".format(sparse_scores))

        ret['semantic_scores'] = sparse_scores.F
        ret['semantic_feats'] = out.F
        ret['sparse_semantic_scores'] = sparse_scores
        ret['sparse_semantic_feats'] = out

        if self.training:
            if labels is None:
                raise ValueError('labels are required during training')
            loss = self.loss_fn(sparse_scores.F, labels)
            ret['loss'] = loss
            if self.config.MODEL.apply_scene_cls:
                scene_scores = self.final_scene(out)
                ret['loss_scene'] = self.loss_scene_fn(scene_scores, labels)

            apply_hack_loss = self.mlp_check_pseudo_instance_label and not self.is_valid_pseudo_instance_label(labels)

            # we only use mlp during training
            if self.mlp is not None:
                mlp_feats = self.mlp(out)

                if apply_hack_loss:
                    ret['mlp_feats'] = None
                    ret['sparse_mlp_feats'] = None
                    # we need to optimize the param in mlp any way, so we do a little hack here
                    ret['loss'] += 1e-6 * torch.nn.functional.smooth_l1_loss(mlp_feats.F,
                                                                             torch.zeros_like(mlp_feats.F).fill_(
                                                                                 torch.mean(mlp_feats.F)))
                else:
                    ret['mlp_feats'] = mlp_feats.F
                    ret['sparse_mlp_feats'] = mlp_feats

            # we only use attentive_pool during training
            POOL_FEATS_KEY = self.config.MODEL.pic_attentive_pool_feats_key
            pic_attentive_pool = self.config.MODEL.pic_attentive_pool
            pic_attentive_pool_with_coord = self.config.MODEL.pic_attentive_pool_with_coord
            assert POOL_FEATS_KEY in ret
            SPARSE_POOL_FEATS_KEY = 'sparse_{}'.format(POOL_FEATS_KEY)
            assert SPARSE_POOL_FEATS_KEY in ret  # in case coord is needed
            if pic_attentive_pool:
                pic_idxes, pic_labels = self._get_pic_idxes(x.C, labels,
                                                            predicted_scores=torch.softmax(
                                                                sparse_scores.F.clone().detach(),
                                                                dim=1),
                                                            return_label=True)
                # acquire feats first
                sparse_att_feats = ret[SPARSE_POOL_FEATS_KEY]  # in case the coords is required
                att_feats = self._get_sampled_pic_feats_by_idxes(feats=sparse_att_feats.F,
                                                                 idxes=pic_idxes)  # (num_pic, num_max_points, feats_dim)
                if pic_attentive_pool_with_coord:
                    # acquire coords if required, coords are always at GPU and type int
                    att_coords = self._get_sampled_pic_feats_by_idxes(feats=sparse_att_feats.C,
                                                                      idxes=pic_idxes)  # (num_pic, num_max_points, coords_dim)
                    att_coords = att_coords[:, :, 1:].float().to(att_feats.device)
                    # logging.info("===> att_coords.size() {}".format(att_coords.size()))
                    # logging.info("===> att_feats.size() {}".format(att_feats.size()))
                    att_feats = torch.cat([att_feats, att_coords], dim=2)
                    # logging.info("===> att_coords[0]: {}".format(att_coords[0]))

                # apply embeddings
                # att_embeds = self.pic_embeddings[pic_labels, :]  # (num_pic, dim)
                # att_embeds = att_embeds.unsqueeze(1)  # (num_pic, dim) => (num_pic, 1, dim)

                # logging.info("===> att_embeds.size() {}".format(att_embeds.size()))
                # logging.info("===> att_feats.size() {}".format(att_feats.size()))

                # concat att_feats and att_embeds
                # att_feats = torch.cat([att_embeds, att_feats],  # (num_pic, 1 + num_max_points, dim)
                #                       dim=1)  # cls embeddings is the first token

                # apply attentive pool
                att_feats = self.pic_attentive_pool_module(att_feats)  # (num_pic, num_max_points, dim)

                # retrieve the first cls embedding
                # pic_feats = att_feats[:, 0, :]  # (num_pic, dim)
                pic_feats = torch.mean(att_feats, dim=1, keepdim=False)  # (num_pic, dim)
                # logging.info("===> pic_feats dim: {}".format(pic_feats.size()))

                if apply_hack_loss:
                    ret['pic_feats'] = None
                    ret['pic_labels'] = None
                    # we need to optimize the param in att pool any way, so we do a little hack here
                    ret['loss'] += 1e-6 * torch.nn.functional.smooth_l1_loss(
                        pic_feats, torch.zeros_like(pic_feats).fill_(torch.mean(pic_feats))
                    )
                else:
                    ret['pic_feats'] = pic_feats
                    ret['pic_labels'] = pic_labels

        return ret

    def _base_forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = MO.cat(out, out_b3p8)
        out = self.block5(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = MO.cat(out, out_b2p4)
        out = self.block6(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = MO.cat(out, out_b1p2)
        out = self.block7(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = MO.cat(out, out_p1)
        out = self.block8(out)

        return out

    def _pic_forward(self, x, labels):
        # logging.info("===> x.size(): {}".format(x.size()))
        pic_idxes, pic_labels = self._get_pic_idxes(x.C, labels, return_label=True)

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        # logging.info("===> out_p1.size(): {}".format(out_p1.size()))

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        # logging.info("===> out_b1p2.size(): {}".format(out_b1p2.size()))

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        # logging.info("===> out_b2p4.size(): {}".format(out_b2p4.size()))

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        # logging.info("===> out_b3p8.size(): {}".format(out_b3p8.size()))

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        # logging.info("===> out4.size(): {}".format(out.size()))

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        # logging.info("===> convtr4p16s2.size(): {}".format(out.size()))

        out = MO.cat(out, out_b3p8)
        out = self.block5(out)
        # logging.info("===> out5: {}".format(out))
        # logging.info("==> out5.size(): {}".format(out.size()))
        # pic block5 here
        # pic_out = self.pic_block5(self._get_pic_feats_by_idxes(feats=out.F, idxes=pic_idxes))

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = MO.cat(out, out_b2p4)
        out = self.block6(out)
        # logging.info("==> out6.size(): {}".format(out.size()))
        # pic block6 here, with residual connection
        # pic_out = self.pic_block6(self._get_pic_feats_by_idxes(feats=out.F, idxes=pic_idxes)) + pic_out

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = MO.cat(out, out_b1p2)
        out = self.block7(out)
        # logging.info("==> out7.size(): {}".format(out.size()))
        # pic block7 here, with residual connection
        # pic_out = self.pic_block7(self._get_pic_feats_by_idxes(feats=out.F, idxes=pic_idxes)) + pic_out

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = MO.cat(out, out_p1)
        out = self.block8(out)
        # logging.info("==> out8.size(): {}".format(out.size()))
        # pic block8 here, with residual connection
        # pic_out = self.pic_block8(self._get_pic_feats_by_idxes(feats=out.F, idxes=pic_idxes)) + pic_out
        pic_out = self.pic_block8(self._get_pic_feats_by_idxes(feats=out.F, idxes=pic_idxes))

        # exit(0)

        return out, pic_out, pic_labels

    def _get_pic_idxes(self, coords, labels, predicted_scores, return_label=False):
        num_group_point = self.config.TRAINER.pseudo_instance_contrastive_num_group_point
        check_label = self.config.TRAINER.pseudo_instance_contrastive_check_label

        predicted_scores = predicted_scores.clone().detach()
        predicted_labels = torch.argmax(predicted_scores, dim=1)  # (N, 1)
        labeled_idx = labels != self.config.DATA.ignore_label
        labeled_coords = coords[labeled_idx, :]

        batch_ids = torch.unique(coords[:, 0])

        batchId2coords = {b_id.item(): coords[coords[:, 0] == b_id, 1:] for b_id in batch_ids}

        pic_idxes = []

        for query_coord, query_label in zip(labeled_coords, labels[labeled_idx]):  # (4,)
            batch_id = query_coord[0].item()
            query_coord = query_coord[1:].unsqueeze(0)  # (1, 3)
            key_coords = batchId2coords[batch_id]  # (M, 3)
            dist = euclidean_distance(query_coord, key_coords)  # (1, M)

            # the closest point is itself, but we use it since avg pooling is applied
            _, closest_index = torch.topk(dist, k=num_group_point, largest=False, dim=1)  # (1, num_group_point)

            if check_label:
                # TODO accelerate
                # new_closest_index = [
                #     _index for _index in closest_index[0] if predicted_labels[_index] == query_label
                # ]

                new_closest_index = closest_index[0][
                    predicted_labels[closest_index[0]] == query_label].numpy().tolist()

                if len(new_closest_index) == 0:
                    # append self if empty
                    # this is reasonable since self have the correct label
                    new_closest_index.append(closest_index[0][0])
                closest_index = torch.tensor([new_closest_index], device=closest_index.device)

            pic_idxes.append(closest_index[0])

        if return_label:
            return pic_idxes, labels[labeled_idx]

        return pic_idxes

    def _get_pic_feats_by_idxes(self, feats, idxes):
        # avg pool is applied TODO, more sophisticated pool
        # logging.info("===> feats.size(0): {}".format(feats.size(0)))
        # logging.info("===> max(idxes): {}".format(torch.max(torch.tensor([torch.max(idx) for idx in idxes]))))
        return torch.stack([torch.mean(feats[idx, :], dim=0) for idx in idxes], dim=0)

    def _get_sampled_pic_feats_by_idxes(self, feats, idxes):
        # TODO apply mask

        num_max_points = max([idx.size(0) for idx in idxes])
        num_pic_feats = len(idxes)
        feats_tensor = torch.zeros((num_pic_feats, num_max_points, feats.size(1)),
                                   dtype=feats.dtype, device=feats.device)
        # logging.info("===> feats.size(): {}".format(feats.size()))
        # logging.info("===> num_max_points: {}".format(num_max_points))
        for pic_id, idx in enumerate(idxes):
            feats_tensor[pic_id, :idx.size(0), :] = feats[idx, :]

        return feats_tensor


@MODEL_REGISTRY.register()
class Res16UNet14(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


@MODEL_REGISTRY.register()
class Res16UNet18(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


@MODEL_REGISTRY.register()
class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


@MODEL_REGISTRY.register()
class Res16UNet50(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


@MODEL_REGISTRY.register()
class Res16UNet101(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


@MODEL_REGISTRY.register()
class Res16UNet14A(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


@MODEL_REGISTRY.register()
class Res16UNet14A2(Res16UNet14A):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


@MODEL_REGISTRY.register()
class Res16UNet14B(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


@MODEL_REGISTRY.register()
class Res16UNet14B2(Res16UNet14B):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


@MODEL_REGISTRY.register()
class Res16UNet14B3(Res16UNet14B):
    LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


@MODEL_REGISTRY.register()
class Res16UNet14C(Res16UNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


@MODEL_REGISTRY.register()
class Res16UNet14D(Res16UNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


@MODEL_REGISTRY.register()
class Res16UNet18A(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


@MODEL_REGISTRY.register()
class Res16UNet18B(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


@MODEL_REGISTRY.register()
class Res16UNet18D(Res16UNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


@MODEL_REGISTRY.register()
class Res16UNet34A(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


@MODEL_REGISTRY.register()
class Res16UNet34B(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


@MODEL_REGISTRY.register()
class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


@MODEL_REGISTRY.register()
class STRes16UNetBase(Res16UNetBase):
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
        super(STRes16UNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


@MODEL_REGISTRY.register()
class STRes16UNet14(STRes16UNetBase, Res16UNet14):
    pass


@MODEL_REGISTRY.register()
class STRes16UNet14A(STRes16UNetBase, Res16UNet14A):
    pass


@MODEL_REGISTRY.register()
class STRes16UNet18(STRes16UNetBase, Res16UNet18):
    pass


@MODEL_REGISTRY.register()
class STRes16UNet34(STRes16UNetBase, Res16UNet34):
    pass


@MODEL_REGISTRY.register()
class STRes16UNet50(STRes16UNetBase, Res16UNet50):
    pass


@MODEL_REGISTRY.register()
class STRes16UNet101(STRes16UNetBase, Res16UNet101):
    pass


@MODEL_REGISTRY.register()
class STRes16UNet18A(STRes16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


@MODEL_REGISTRY.register()
class STResTesseract16UNetBase(STRes16UNetBase):
    CONV_TYPE = ConvType.HYPERCUBE


@MODEL_REGISTRY.register()
class STResTesseract16UNet18A(STRes16UNet18A, STResTesseract16UNetBase):
    pass


if __name__ == '__main__':
    def run_Res16UNet34C():
        import MinkowskiEngine as ME
        from util.config import CfgNode

        p = '/home/liulizhao/projects/WeaklySegmentationKit/third_party/contrastive_scene_contexts/downstream/semseg/config/default.yaml'
        cfg = CfgNode(CfgNode.load_yaml_with_base(p))
        model = Res16UNet34C(in_channels=3, out_channels=20, config=cfg)
        print("===> model: \n{}".format(model))

        num_p = 100

        coords = torch.randint(low=0, high=100, size=(num_p, 4)).int()
        feats = torch.randn((num_p, 3))

        _inputs = ME.SparseTensor(feats, coords)

        print("===> _inputs: \n{}".format(_inputs))

        print(type(_inputs))
        assert isinstance(_inputs, ME.SparseTensor)

        # conduct forward pass
        _output1, _output2 = model(_inputs)
        print("===> _output1: \n{}\n\n".format(_output1))
        print("===> _output2: \n{}".format(_output2))

        print("===> _output1.feats.size(): {}".format(_output1.feats.size()))
        print("===> _output2.feats.size(): {}".format(_output2.feats.size()))


    run_Res16UNet34C()