import logging
import math
import random
import pickle
import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

from meta_data.constant import PROJECT_NAME


class MaxPointsSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        data_source (dataset): The source of data.
        max_point (int): the max points that a batch data contain.
        shuffle (bool): if ``True``, the sampler shuffle the data_source
        cache_path (str): It takes a lot of time to get points numbers,
            if we cache it, we can save some time
        clear_cache (bool): If ``True``, clear the cache file and reload
            data points numbers
    """

    def __init__(self, data_source: Dataset, max_point, shuffle, cache_path, clear_cache=False, seed=123):
        super().__init__(data_source)
        if not isinstance(max_point, int) or isinstance(max_point, bool) or \
                max_point <= 0:
            raise ValueError("max_point should be a positive integer value, "
                             "but got batch_size={}".format(max_point))
        self.data_source = data_source
        self.max_point = max_point
        self.shuffle = shuffle
        self.data_length = len(data_source)
        self.visited = [0 for _ in range(self.data_length)]
        self.cache_path = cache_path
        self.batch_list_length = None
        self.epoch = 0
        self.seed = seed

        # if it is the first time to load, we need to load all data here, then
        # we write in a file, so we can read the file to get all points numbers next time
        if clear_cache or not os.path.exists(cache_path):
            self.points_numbers_dict = {i: self.get_point_numbers(i) for i in range(self.data_length)}
            with open(cache_path, 'wb') as f:
                pickle.dump(self.points_numbers_dict, f)
        else:
            with open(cache_path, 'rb') as f:
                self.points_numbers_dict = pickle.load(f)

        self.sorted_list = sorted(self.points_numbers_dict.items(), key=lambda k: k[1])
        # self.sorted_list = [(k, v) for k, v in self.points_numbers_dict.items()]
        self.avg_numbers = sum([item[1] for item in self.sorted_list]) // len(self.sorted_list)

        assert self.max_point >= self.sorted_list[-1][1], \
            "max_point: {} should greater than the maximum number of data point: {}".format(
                self.max_point, self.sorted_list[-1][1]
            )

        self.batch_list_length, self.batch_list = self.batch_sample()

    # TODO the last batch may not have enough points, consider fill in with replica data
    def batch_sample(self):
        point_numbers = 0
        batch = []
        batch_list = []

        # left_side contains little points of data
        # right_side contains large points of data
        _half = len(self.sorted_list) // 2
        left_side = self.sorted_list[:_half]
        right_side = self.sorted_list[_half:]

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            left_indices = torch.randperm(len(left_side), generator=g).tolist()
            right_indices = torch.randperm(len(right_side), generator=g).tolist()

            left_side = [left_side[_] for _ in left_indices]
            right_side = [right_side[_] for _ in right_indices]

        # if not shuffles, pair the largest data and the least data one by one
        # if shuffles, shuffle left_side and right_side respectively, and pair them randomly
        right_side = right_side[::-1]
        # contain
        data_list = [number for item in zip(left_side, right_side) for number in item]

        # in case the len(data_list) is odd number, the last value will be ignored, we retain it
        if len(right_side) > len(left_side):
            data_list.append(right_side[-1])

        # collect index of data_course in batch, making sum of points in the batch will not beyond max_points
        while sum(self.visited) != len(self.visited):
            for index in range(len(self.visited)):
                if self.visited[data_list[index][0]] == 0 and point_numbers + data_list[index][1] <= self.max_point:
                    batch.append(data_list[index][0])
                    point_numbers += data_list[index][1]
                    self.visited[data_list[index][0]] = 1

            actual_maxpoint = sum([self.points_numbers_dict[_] for _ in batch])

            assert 0 < actual_maxpoint <= self.max_point, 'actual_maxpoint = {}'.format(actual_maxpoint)
            batch_list.append(batch)
            batch = []
            point_numbers = 0

        assert list(range(len(self.data_source))) == sorted([i for item in batch_list for i in item])
        self.visited = [0 for _ in range(self.data_length)]
        self.epoch += 1

        # filter batch_list with only 1 datapoint
        batch_list = [_ for _ in batch_list if len(_) > 1]

        return len(batch_list), batch_list

    def __iter__(self):
        yield from self.batch_list
        self.batch_list_length, self.batch_list = self.batch_sample()

    def __len__(self):
        return self.batch_list_length

    def load_cache(self, path):
        pass

    # TODO use multi process to load get point numbers
    def get_point_numbers(self, index) -> int:
        return self.data_source[index][2].shape[0]

    # find the result that just less or equal than target
    def binary_search(self, target: int) -> Tuple[int, int]:
        result = 0
        index = 0

        unvisited_value_list = [self.sorted_list[_] for _ in range(len(self.visited)) if self.visited[_] == 0]
        left, right = 0, len(unvisited_value_list) - 1

        while left <= right:
            mid = left + (right - left) // 2
            if unvisited_value_list[mid][1] > target:
                right = mid - 1
            else:
                left = mid + 1

        for i in reversed(range(left)):
            if unvisited_value_list[i][1] <= target:
                result = unvisited_value_list[i][1]
                index = unvisited_value_list[i][0]
                break
        return index, result


class DistributedMaxPointsSampler:
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        data_source (dataset): The source of data.
        max_point (int): the max points that a batch data contain.
        shuffle (bool): if ``True``, the sampler shuffle the data_source
        cache_path (str): It takes a lot of time to get points numbers,
            if we cache it, we can save some time
        clear_cache (bool): If ``True``, clear the cache file and reload
            data points numbers
    """

    def __init__(self, data_source: Dataset, max_point, shuffle, cache_path, clear_cache: bool = False,
                 seed: int = 123, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, drop_last: bool = False):
        if not isinstance(max_point, int) or isinstance(max_point, bool) or \
                max_point <= 0:
            raise ValueError("max_point should be a positive integer value, "
                             "but got batch_size={}".format(max_point))
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.data_source = data_source
        self.shuffle = shuffle
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.seed = seed
        self._sampler = MaxPointsSampler(data_source, max_point, shuffle, cache_path, clear_cache, seed)

        self.batch_list_length, self.batch_list, self.num_batch_samples, self.total_batch_size = self.batch_sample()

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.batch_list_length, generator=g).tolist()
        else:
            indices = list(range(self.batch_list_length))

        if not self.drop_last:
            padding_batch_size = self.total_batch_size - self.batch_list_length
            if padding_batch_size <= self.batch_list_length:
                indices += indices[:padding_batch_size]
            else:
                indices += (indices * math.ceil(padding_batch_size / self.batch_list_length))[:padding_batch_size]
        else:
            indices = indices[:self.total_batch_size]
        assert len(indices) == self.total_batch_size, "total len(indices): {}, total_batch_size: {}, not equal".format(
            len(indices), self.total_batch_size)

        # subsample
        indices = indices[self.rank:self.total_batch_size:self.num_replicas]
        assert len(
            indices) == self.num_batch_samples, "subsample len(indices): {}, num_batch_samples: {}, not equal".format(
            len(indices), self.num_batch_samples)

        yield from list(map(lambda i: self.batch_list[i], indices))

        self.batch_list_length, self.batch_list, self.num_batch_samples, self.total_batch_size = self.batch_sample()

    def __len__(self):
        return self.num_batch_samples

    def batch_sample(self):
        batch_list_length, batch_list = self._sampler.batch_sample()

        if self.drop_last and len(self.data_source) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to the nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            num_batch_samples = math.ceil(
                (batch_list_length - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            num_batch_samples = math.ceil(batch_list_length / self.num_replicas)  # type: ignore[arg-type]
        total_batch_size = num_batch_samples * self.num_replicas
        self.epoch += 1

        return batch_list_length, batch_list, num_batch_samples, total_batch_size
