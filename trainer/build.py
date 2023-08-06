import logging

from fvcore.common.registry import Registry

from meta_data.constant import PROJECT_NAME

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.__doc__ = """
Registry for trainer that creates for control experiment.
The registered object will be called with `obj(cfg)`.
"""

# from . import otoc_trainer  # noqa F401 isort:skip
# from . import otoc_stage2_trainer  # noqa F401 isort:skip
# from . import otoc_semi_trainer  # noqa F401 isort:skip
from . import fully_supervised_trainer  # noqa F401 isort:skip


def build_trainer(cfg):
    """
    Build a trainer from cfg.TRAINER.name
    """
    logger = logging.getLogger(PROJECT_NAME)
    logger.info('Using trainer {}'.format(cfg.TRAINER.name))
    return TRAINER_REGISTRY.get(cfg.TRAINER.name)(cfg)
