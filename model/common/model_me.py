# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from MinkowskiEngine import MinkowskiNetwork
import logging


class Model(MinkowskiNetwork):
    """
    Base network for all sparse convnet

    By default, all networks are segmentation networks.
    """
    OUT_PIXEL_DIST = -1

    def __init__(self, config):
        super(Model, self).__init__(config.MODEL.D)
        self.in_channels = config.MODEL.in_channels
        self.out_channels = config.MODEL.out_channels
        self.D = config.MODEL.D
        self.config = config

        logging.info("Building model with in_channels={}, out_channels={}, D={}".format(
            self.in_channels, self.out_channels, self.D))


class HighDimensionalModel(Model):
    """
    Base network for all spatio (temporal) chromatic sparse convnet
    """

    def __init__(self, config):
        D = config.MODEL.D
        assert D > 4, "Num dimension smaller than 5"
        super(HighDimensionalModel, self).__init__(config)
