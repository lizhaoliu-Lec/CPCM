import logging

from fvcore.common.registry import Registry

from meta_data.constant import PROJECT_NAME

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets that are created for weak segmentation.
The registered object will be called with `obj(cfg)`.
"""

# from . import otoc_scannetv2  # noqa F401 isort:skip
# from . import semi_scannetv2  # noqa F401 isort:skip
from .fully_supervised import scannet  # noqa F401 isort:skip
from .fully_supervised import stanford  # noqa F401 isort:skip
from .fully_supervised import semantic_kitti  # noqa F401 isort:skip


def build_dataset(cfg):
    """
    Build a dataset from cfg.DATASET.name
    """
    logger = logging.getLogger(PROJECT_NAME)
    logger.info('Using dataset {}'.format(cfg.DATA.name))
    return DATASET_REGISTRY.get(cfg.DATA.name)(cfg)
