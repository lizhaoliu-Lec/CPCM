# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiReLU
from MinkowskiEngine.MinkowskiNormalization import MinkowskiBatchNorm

from model.common.model_me import Model
from model.common.base import ConvType, NormType, get_norm, conv, sum_pool
from model.common.resnet_block import BasicBlock, Bottleneck
from .build import MODEL_REGISTRY


class ResNetBase(Model):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    OUT_PIXEL_DIST = 32
    HAS_LAST_BLOCK = False
    CONV_TYPE = ConvType.HYPERCUBE

    def __init__(self, config):
        if not hasattr(self.config, 'D'):
            self.config.MODEL.D = 3

        D = self.config.MODEL.D
        self.entropy_weight = None
        entropy_weight_type = self.config.MODEL.entropy_weight_type
        if entropy_weight_type:
            logging.info("Using entropy_weight_type={} for cross entropy loss".format(entropy_weight_type))
            if not hasattr(self.config.MODEL, '{}'.format(entropy_weight_type)):
                raise ValueError('Not supported entropy_weight_type: {}'.format(entropy_weight_type))
            self.entropy_weight = getattr(self.config.MODEL, '{}'.format(entropy_weight_type))
            self.entropy_weight = torch.tensor(self.entropy_weight)

        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0

        super(ResNetBase, self).__init__(config)

        self.network_initialization(self.in_channels, self.out_channels, config, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, config, D):

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        dilations = config.MODEL.dilations
        bn_momentum = config.MODEL.bn_momentum
        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.MODEL.conv1_kernel_size, 1),
            stride=1,
            D=D)

        self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[0], 1))
        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[1], 1))
        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[2], 1))
        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[3], 1))

        self.final = conv(
            self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    norm_type=NormType.BATCH_NORM,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D),
                get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
            )
        layers = [block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final(x)

        ret = {}

        if self.training:
            if labels is None:
                raise ValueError('labels are required during training')
            loss = self.loss_fn(x.F, labels)
            ret['loss'] = loss

        ret['semantic_scores'] = x.F
        ret['sparse_semantic_scores'] = x

        return ret

    def loss_fn(self, scores, labels):
        # scores: (N, nClass), float32, cuda
        # labels: (N), long, cuda
        if self.entropy_weight is not None:
            if self.entropy_weight.device != scores.device:
                self.entropy_weight = self.entropy_weight.to(scores.device)
            return F.cross_entropy(scores, labels,
                                   weight=self.entropy_weight,
                                   ignore_index=self.config.DATA.ignore_label)
        else:
            return nn.CrossEntropyLoss(ignore_index=self.config.DATA.ignore_label)(scores, labels)

    def loss_scene_fn(self, scores, labels):
        # scores: point-wise scores (N, C)
        # labels: point-wise labels (N, C), with 255 as ignore label

        num_classes = self.out_channels

        # scene-wise multi-label classification
        batch_ids = torch.unique(scores.C[:, 0])
        batchId2scores = {b_id.item(): scores.F[scores.C[:, 0] == b_id, :] for b_id in batch_ids}
        batchId2labels = {b_id.item(): labels[scores.C[:, 0] == b_id] for b_id in batch_ids}

        batch_scene_scores = []
        batch_scene_labels = []

        for batch_id in batch_ids:
            scene_scores = batchId2scores[batch_id.item()]
            scene_labels = batchId2labels[batch_id.item()]

            # apply scene-wise pooling
            scene_score = torch.mean(scene_scores, dim=0, keepdim=False)

            # get scene label (one hot)
            # remove the ignore label
            scene_label = torch.unique(scene_labels[scene_labels != self.config.DATA.ignore_label])
            scene_label_one_hot = torch.zeros(num_classes).long().to(labels.device)
            scene_label_one_hot[scene_label] = 1

            # logging.info("===> scene_score.size(): {}".format(scene_score.size()))
            # logging.info("===> scene_label_one_hot.size(): {}".format(scene_label_one_hot.size()))

            assert scene_score.size() == scene_label_one_hot.size()

            batch_scene_scores.append(scene_score)
            batch_scene_labels.append(scene_label_one_hot)

        scene_scores = torch.stack(batch_scene_scores, dim=0)  # (bs, num_classes)
        scene_labels = torch.stack(batch_scene_labels, dim=0)  # (bs, num_classes)

        # logging.info("===> scene_scores.size(): {}".format(scene_scores.size()))
        # logging.info("===> scene_labels.size(): {}".format(scene_labels.size()))
        #
        # logging.info("===> scene_scores: {}".format(scene_scores))
        # logging.info("===> scene_labels: {}".format(scene_labels))

        assert scene_scores.size() == scene_labels.size()

        return F.multilabel_soft_margin_loss(scene_scores, scene_labels)


@MODEL_REGISTRY.register()
class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


@MODEL_REGISTRY.register()
class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


@MODEL_REGISTRY.register()
class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


@MODEL_REGISTRY.register()
class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


@MODEL_REGISTRY.register()
class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


class STResNetBase(ResNetBase):
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, config):
        if not hasattr(self.config.MODEL, 'D'):
            self.config.MODEL.D = 4
        super(STResNetBase, self).__init__(config)


@MODEL_REGISTRY.register()
class STResNet14(STResNetBase, ResNet14):
    pass


@MODEL_REGISTRY.register()
class STResNet18(STResNetBase, ResNet18):
    pass


@MODEL_REGISTRY.register()
class STResNet34(STResNetBase, ResNet34):
    pass


@MODEL_REGISTRY.register()
class STResNet50(STResNetBase, ResNet50):
    pass


@MODEL_REGISTRY.register()
class STResNet101(STResNetBase, ResNet101):
    pass


@MODEL_REGISTRY.register()
class STResTesseractNetBase(STResNetBase):
    CONV_TYPE = ConvType.HYPERCUBE


@MODEL_REGISTRY.register()
class STResTesseractNet14(STResTesseractNetBase, STResNet14):
    pass


@MODEL_REGISTRY.register()
class STResTesseractNet18(STResTesseractNetBase, STResNet18):
    pass


@MODEL_REGISTRY.register()
class STResTesseractNet34(STResTesseractNetBase, STResNet34):
    pass


@MODEL_REGISTRY.register()
class STResTesseractNet50(STResTesseractNetBase, STResNet50):
    pass


@MODEL_REGISTRY.register()
class STResTesseractNet101(STResTesseractNetBase, STResNet101):
    pass
