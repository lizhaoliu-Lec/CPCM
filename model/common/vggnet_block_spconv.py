import torch.nn as nn

from spconv import SparseModule, SparseSequential
from spconv import SubMConv3d


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(inplace=True),
            SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, _input):
        return self.conv_layers(_input)
