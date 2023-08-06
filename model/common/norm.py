import functools
import torch.nn as nn


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels: int, channels_per_groups: int, eps: float = 1e-5, affine: bool = True) -> None:
        num_groups = int(num_channels / channels_per_groups)
        if num_groups == 0: num_groups = 1
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


def get_norm_fn(norm_fn_name, eps, momentum, channels_per_groups):
    if norm_fn_name == "BN":
        return functools.partial(nn.BatchNorm1d, eps=eps, momentum=momentum)
    if norm_fn_name == "LN":
        return functools.partial(nn.LayerNorm, eps=eps, )
    if norm_fn_name == "GN":
        return functools.partial(GroupNorm, channels_per_groups=channels_per_groups, eps=eps)
    if norm_fn_name == "IN":
        return functools.partial(nn.InstanceNorm1d, eps=eps, momentum=momentum)
