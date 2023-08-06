import argparse
from typing import List

import yaml
import os

from util.utils import get_time_string
from fvcore.common.config import CfgNode as _CfgNode, BASE_KEY
from yacs.config import _assert_with_logging, _check_and_coerce_cfg_value_type


class CfgNode(_CfgNode):
    pass

    # commnet out for valid field check

    # def merge_from_list(self, cfg_list: List[str]):
    #     """
    #     Args:
    #         cfg_list (list): list of configs to merge from.
    #     """
    #     keys = set(cfg_list[0::2])
    #     assert (
    #             BASE_KEY not in keys
    #     ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
    #     """Merge config (keys, values) in a list (e.g., from command line) into
    #             this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
    #             """
    #     _assert_with_logging(
    #         len(cfg_list) % 2 == 0,
    #         "Override list has odd length: {}; it must be a list of pairs".format(
    #             cfg_list
    #         ),
    #     )
    #     root = self
    #     for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
    #         if root.key_is_deprecated(full_key):
    #             continue
    #         if root.key_is_renamed(full_key):
    #             root.raise_key_rename_error(full_key)
    #         key_list = full_key.split(".")
    #         d = self
    #         for subkey in key_list[:-1]:
    #             if subkey not in d:  # hack to allow new key
    #                 d[subkey] = CfgNode()
    #             d = d[subkey]
    #         subkey = key_list[-1]
    #
    #         value = self._decode_cfg_value(v)
    #
    #         if subkey not in d:  # hack to allow new key
    #             d[subkey] = value
    #
    #         value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
    #         d[subkey] = value


def get_exp_path(base_dir, config_filepath, exp_id=None):
    config_filename = config_filepath.config.split('/')[-1][:-5]  # last ele is filename, subtracting .yaml
    time_string = get_time_string()
    if exp_id is not None:
        last_dir = '{}.{}.{}'.format(config_filename, exp_id, time_string)
    else:
        last_dir = '{}.{}'.format(config_filename, time_string)

    return os.path.join(base_dir, last_dir)


def get_parser():
    parser = argparse.ArgumentParser(description='Weakly Segmentation Kit')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml',
                        help='path to config file')

    # pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    # exp
    parser.add_argument('--exp-dir', type=str, default='exp', help='dir for save experiment files')
    parser.add_argument('--exp-id', type=str, default=None, help='sub dir under the base dir '
                                                                 'for saving experiment files')
    parser.add_argument('--debug', type=bool, default=False, help='debug the code or not, affect the logger')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    exp_path = get_exp_path(os.path.join(args_cfg.exp_dir, args_cfg.dataset, args_cfg.model_name),
                            args_cfg.config,
                            args_cfg.exp_id)
    args_cfg.exp_path = exp_path

    return args_cfg


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='Weakly Segmentation Kit')
        parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml',
                            help='path to config file')

        # pretrain
        parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

        # exp
        parser.add_argument('--exp-dir', type=str, default='exp', help='dir for save experiment files')
        parser.add_argument('--exp-id', type=str, default=None, help='sub dir under the base dir '
                                                                     'for saving experiment files')
        parser.add_argument('--debug', type=bool, default=False, help='debug the code or not, affect the logger')

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

        args_cfg = parser.parse_args()
        return args_cfg


    def run_config():
        args_cfg = parse_args()
        for key, value in vars(args_cfg).items():
            print(key, value)
        print("===> args_cfg: \n{}".format(args_cfg))
        cfg = CfgNode(
            CfgNode.load_yaml_with_base('/home/liulizhao/projects/WeaklySegmentationKit/config.exp/otoc_semi_mse.yaml'),
        )
        print("===> cfg1: \n{}".format(cfg))
        print("===> cfg1 DATA.name: \n{}".format(cfg.DATA.name))
        cfg.merge_from_list(args_cfg.opts)
        print("===> cfg2 DATA.name: \n{}".format(cfg.DATA.name))
        print("===> cfg2: \n{}".format(cfg))

        cfg.aaaaaaaaaaaaaaaaaa = 1000000000000
        print("===> cfg3: \n{}".format(cfg))


    run_config()
