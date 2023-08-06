import torch.distributed as dist
import torch


def reduce_tensor(tensor: torch.Tensor, size):
    _tensor = tensor.clone()
    dist.all_reduce(_tensor, op=dist.ReduceOp.SUM)
    _tensor /= size
    return _tensor


