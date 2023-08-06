"""
Reference: https://github.com/liuzhengzhe/One-Thing-One-Click/tree/master/3D-U-Net/util
"""

import logging
import os
import sys

from meta_data.constant import PROJECT_NAME, PROJECT_ABBREV_NAME
from meta_data.logging import LOG_FORMAT
from util.utils import get_time_string

from termcolor import colored

__all__ = ['setup_logger', 'config_logger']


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name")
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        # print("===> self._root_name, self._abbrev_name: {}, {}".format(self._root_name, self._abbrev_name))
        # print("===> before record.name: {}".format(record.name))
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        # print("===> after record.name: {}".format(record.name))
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def config_logger(log_file=None, rank=0, debug=False):
    # LOG_FORMAT = '[%(name)s %(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s'
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s %(pathname)s %(funcName)s]: ", "green") +
        colored("\n%(levelname)s: [RANK_{}]".format(rank), "blue") + " %(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name=PROJECT_NAME,
        abbrev_name=PROJECT_ABBREV_NAME,
    )
    basic_formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(pathname)s %(funcName)s]: \n%(levelname)s: [RANK_{}] %(message)s".format(rank),
        datefmt="%m/%d %H:%M:%S"
    )

    # add stream handler for screen output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)  # screen output can be configured

    # add file handler for file output
    file_handler = None
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(basic_formatter)
        file_handler.setLevel(logging.DEBUG)  # file output is always debug, for maintaining all information

    wsk_logger = logging.getLogger(PROJECT_NAME)
    wsk_logger.setLevel(logging.DEBUG)  # set global log level
    """Fix bug: See: https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    Note: If you attach a handler to a logger and one or more of its ancestors, it may emit the same 
    record multiple times. In general, you should not need to attach a handler to more than one logger - if you just 
    attach it to the appropriate logger which is highest in the logger hierarchy, then it will see all events logged 
    by all descendant loggers, provided that their propagate setting is left set to True. A common scenario is to 
    attach handlers only to the root logger, and to let propagation take care of the rest. 
    """
    # wsk_logger.addHandler(stream_handler)
    # wsk_logger.addHandler(file_handler)

    # also config root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stream_handler)
    if file_handler is not None:
        root_logger.addHandler(file_handler)
    # do not set propagate = False, we need this feature for easy calling logging.info, logging.debug etc.

    return wsk_logger


def _get_log_filepath(_cfg):
    time_string = get_time_string()

    if _cfg.task == 'train':
        log_dir = _cfg.exp_path
        log_filename = 'train.{}.log'.format(time_string)
    elif _cfg.task == 'test':
        log_dir = os.path.join(
            _cfg.exp_path,
            'result.epoch.{}.split.{}'.format(_cfg.test_epoch, _cfg.split))
        log_filename = 'test.{}.log'.format(time_string)
    else:
        raise ValueError('Unsupported task found `{}`'.format(_cfg.task))

    return log_dir, log_filename


def setup_logger(_cfg):
    # log_dir, log_filename = _get_log_filepath(_cfg)
    # log_filepath = os.path.join(log_dir, log_filename)
    # if not os.path.exists(log_dir):
    #     os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    log_filepath = os.path.join(_cfg.exp_path, '{}-log.txt'.format(_cfg.log_time))
    config_logger(log_filepath, _cfg.rank, _cfg.GENERAL.debug)
    _logger = logging.getLogger(PROJECT_NAME)
    _logger.info('Log will be synced to file: ** {} **'.format(log_filepath))


def get_logger(name=PROJECT_NAME):
    return logging.getLogger(name=name)


if __name__ == '__main__':
    def run_create_logger():
        # LL = config_logger('log_debug.txt', rank=0, debug=False)
        LL = config_logger()
        logger = logging.getLogger(PROJECT_NAME)
        logger.info('hehehehheeeeeeeeeee')
        logger.debug('dddddddddddddddddd')
        LL.info('hhhhhh')
        # logger.debug('jijijiji')
        # LL.debug('JJJJJJJJJJJ')
        logging.info("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        logging.info("AAAAAAAAAAAAAA")
        logging.debug("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")


    run_create_logger()
