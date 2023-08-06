import os
import yaml
import logging
import argparse

import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch

from meta_data.constant import PROJECT_NAME
from trainer import build_trainer
from util.config import CfgNode
from util.log import setup_logger
from util.utils import recursively_set_attr, create_exp_path
import mmap  # noqa F401 isort:skip


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description='Parser for specifying WeaklySegmentationKit (WSK) config path')

    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--log_time", type=str, required=True)

    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
                    """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def setup(local_rank, local_world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=local_rank, world_size=local_world_size)


def cleanup():
    dist.destroy_process_group()


def main():
    torch.backends.cudnn.enabled = False  # for compatible with 3090

    args = get_arguments()
    config_path = args.config

    # init process group
    setup(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))

    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }

    # with open(config_path, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    config = CfgNode(CfgNode.load_yaml_with_base(config_path))
    config.merge_from_list(args.opts)

    # set config
    config.config_path = config_path
    config.local_rank = args.local_rank
    config.rank = dist.get_rank()
    config.world_size = dist.get_world_size()
    config.log_time = args.log_time
    config.is_distributed = True

    # max_factor = min(config.world_size // 2, 3)
    max_factor = config.world_size

    config.OPTIMIZER.lr = config.OPTIMIZER.lr * max_factor

    create_exp_path(config)

    # set logger
    setup_logger(config)
    logger = logging.getLogger(PROJECT_NAME)

    logger.info(
        "Initializing process group: MASTER_ADDR: {}, MASTER_PORT: {}, RANK: {}, WORLD_SIZE: {}, BACKEND: {}".format(
            env_dict["MASTER_ADDR"], env_dict["MASTER_PORT"], config.rank, env_dict["WORLD_SIZE"], dist.get_backend()))
    logger.debug("Using configuration: \n{}".format(config))

    logger.info("Multiply lr by max_factor: {}, resulting in lr = {}".format(max_factor, config.OPTIMIZER.lr))

    torch.cuda.set_device(args.local_rank)

    # build trainer and train the model
    trainer = build_trainer(cfg=config)
    trainer.train()
    # trainer.test_for_scannet()

    # Tear down the process group
    cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        # Tear down the process group
        cleanup()
