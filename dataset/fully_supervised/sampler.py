import logging

import math

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

      Arguments:
          data_source (Dataset): dataset to sample from
      """

    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = list(range(len(self.data_source)))
        if self.shuffle:
            perm = torch.randperm(len(self.data_source)).tolist()
        self._perm = perm

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
            raise StopIteration()

        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

    next = __next__  # Python 2 compatibility


class DistributedInfSampler(InfSampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=True):
        super(DistributedInfSampler, self).__init__(data_source, shuffle)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        logging.info("Using num_replicas={}, rank={}, shuffle={} for distributed sampler".format(
            num_replicas, rank, shuffle
        ))

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.it = 0
        self.num_samples = int(math.ceil(len(self.data_source) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.reset_permutation()

    def __next__(self):
        it = self.it * self.num_replicas + self.rank
        value = self._perm[it % len(self._perm)]
        self.it = self.it + 1

        if (self.it * self.num_replicas) >= len(self._perm):
            self.reset_permutation()
            self.it = 0
            raise StopIteration()
        return value

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    def run_DistributedInfSampler():
        sam = DistributedInfSampler(list(range(100)), num_replicas=2, rank=0, shuffle=True)
        for data in sam:
            print(data)
        print("===> len(sam): {}".format(len(sam)))

        # for i in sam:
        #     print(i)


    def run_InfSampler():
        sam = InfSampler(list(range(100)), shuffle=True)
        for data in sam:
            print(data)
        print("===> len(sam): {}".format(len(sam)))


    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # run_DistributedInfSampler()
    run_InfSampler()
