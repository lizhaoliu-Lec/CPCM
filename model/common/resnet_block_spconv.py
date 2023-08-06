import torch.nn as nn

from spconv import SparseModule, SparseSequential
from spconv import SubMConv3d
from spconv import SparseConvTensor


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = SparseSequential(
                SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(inplace=True),
            SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(inplace=True),
            SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, _input):
        identity = SparseConvTensor(_input.features, _input.indices, _input.spatial_shape, _input.batch_size)

        output = self.conv_branch(_input)
        # output = output.replace_feature(output.features + self.i_branch(identity).features)
        output.features += self.i_branch(identity).features

        return output
