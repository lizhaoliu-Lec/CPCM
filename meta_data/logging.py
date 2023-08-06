import logging

LOG_FORMAT = '[%(name)s %(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s'


def create_debug_logger():
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)  # filename: build a FileHandler


def create_info_logger():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)  # filename: build a FileHandler
