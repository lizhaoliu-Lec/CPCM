import logging
from bisect import bisect_right
from collections import Counter
from functools import wraps

import warnings
import weakref

import types

from meta_data.constant import PROJECT_NAME
from torch.optim.optimizer import Optimizer

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

SAVE_STATE_WARNING = "Please also save or load the state of the optimizer when saving or loading the scheduler."


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/1.7.1/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/1.7.1/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class _CompatibleLRScheduler(_LRScheduler):
    """
    _CompatibleLRScheduler that can cope with stepping LR in both epoch and step granularity.
    Also provides additionally warm up before using the actually lr scheduler.
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False, step_by='epoch', warm_step=0, warm_step_by='step'):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)
        logger = logging.getLogger(PROJECT_NAME)
        assert step_by in ['epoch', 'step'], 'got unexpected value for step_by `{}`'.format(step_by)
        assert warm_step_by in ['epoch', 'step'], 'got unexpected value for warm_step_by `{}`'.format(warm_step_by)
        self.step_by = step_by
        self.warm_step_by = warm_step_by
        self.warm_step = warm_step
        self.warm_start_lr = 1e-6
        self.warm_step_count = 0
        # record the expected lr from optimizer for warmup
        self.warm_end_lrs = [group['lr'] for group in self.optimizer.param_groups]

        if warm_step_by == 'epoch':
            logger.warning('Warmup by epoch requires many epochs to warmup, maker sure of that.')
        if warm_step_by == 'epoch' and warm_step == 1:
            logger.warning('Warmup has no effect when step_by={} and warm_step={}'.format(step_by, warm_step))

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.warm_step_count = 0
        if self.warm_step != 0:
            self.warmup_step()

    def step(self, epoch=None, step_type=None):
        if step_type is None:
            super().step(epoch=epoch)  # for backward compatibility
            return

        assert step_type in ['epoch', 'step'], 'got unexpected value for step_type `{}`'.format(step_type)

        # (1) warmup first
        if self.warm_step_count < self.warm_step and step_type == self.warm_step_by:
            self.warmup_step()
            return

        # (2) lr scheduler after warmup
        if step_type == self.step_by:
            super().step(epoch=epoch)
            return

    def warmup_step(self):
        for param_group, warmup_end_lr in zip(self.optimizer.param_groups, self.warm_end_lrs):
            param_group['lr'] = self.compute_warm_up_lr(warmup_end_lr)

        self.warm_step_count += 1

    def compute_warm_up_lr(self, warmup_end_lr):
        warm_step = self.warm_step
        assert warm_step != 0, 'warm_step = {}'.format(warm_step)

        x1, y1 = 0, self.warm_start_lr
        x2, y2 = self.warm_step, warmup_end_lr
        k = (y2 - y1) / (x2 - x1)
        b = self.warm_start_lr
        return self.warm_step_count * k + b


class StepLR(_CompatibleLRScheduler):
    """Decays the learning rate of each parameter group by gamma every
        step_size epochs. Notice that such decay can happen simultaneously with
        other changes to the learning rate from outside this scheduler. When
        last_epoch=-1, sets initial lr as lr.
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
        Example:
            >>> # Assuming optimizer uses lr = 0.05 for all groups
            >>> # lr = 0.05     if epoch < 30
            >>> # lr = 0.005    if 30 <= epoch < 60
            >>> # lr = 0.0005   if 60 <= epoch < 90
            >>> # ...
            >>> # xdoctest: +SKIP
            >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step()
        """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False, **kwargs):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch, verbose, **kwargs)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class MultiStepLR(_CompatibleLRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> # xdoctest: +SKIP
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False, **kwargs):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch, verbose, **kwargs)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class ConstantLR(_CompatibleLRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> # xdoctest: +SKIP
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False, **kwargs):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_iters = total_iters
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose, **kwargs)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters or
                (self.last_epoch != self.total_iters)):
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch == self.total_iters:
            return [group['lr'] * (1.0 / self.factor) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
                for base_lr in self.base_lrs]


class LambdaLR(_CompatibleLRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> # xdoctest: +SKIP
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False, step_by='step',
                 warm_step=0, warm_step_by='step'):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(LambdaLR, self).__init__(optimizer, last_epoch, verbose, step_by=step_by,
                                       warm_step=warm_step, warm_step_by=warm_step_by)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


class LambdaStepLR(LambdaLR):

    def __init__(self, optimizer, lr_lambda, last_step=-1, step_by='step',
                 warm_step=0, warm_step_by='step'):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step, step_by=step_by,
                                           warm_step=warm_step, warm_step_by=warm_step_by)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1, step_by='step',
                 warm_step=0, warm_step_by='step'):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_epoch,
                                     step_by=step_by, warm_step=warm_step, warm_step_by=warm_step_by)


class SquaredLR(LambdaStepLR):
    """ Used for SGD Lars"""

    def __init__(self, optimizer, max_iter, last_epoch=-1, step_by='step',
                 warm_step=0, warm_step_by='step'):
        super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** 2, last_epoch,
                                        step_by=step_by, warm_step=warm_step, warm_step_by=warm_step_by)


class ExpLR(LambdaStepLR):

    def __init__(self, optimizer, step_size, gamma=0.9, last_epoch=-1, step_by='step',
                 warm_step=0, warm_step_by='step'):
        # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
        # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
        # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
        super(ExpLR, self).__init__(optimizer, lambda s: gamma ** (s / step_size), last_epoch, step_by=step_by,
                                    warm_step=warm_step, warm_step_by=warm_step_by)


def build_lr_scheduler(cfg, optimizer, last_epoch=-1):
    logger = logging.getLogger(PROJECT_NAME)
    logger.info('Using lr scheduler {}'.format(cfg.SCHEDULER.name))
    params = dict(vars(cfg.SCHEDULER))
    params.pop('name')
    params['last_epoch'] = last_epoch
    logger.info('LR Scheduler params: {}'.format(params))
    lr_scheduler_factory = {
        'Step': StepLR,
        'MultiStepLR': MultiStepLR,
        'ConstantLR': ConstantLR,
    }
    if cfg.SCHEDULER.name not in lr_scheduler_factory:
        raise ValueError('Unsupported lr scheduler `{}`'.format(cfg.SCHEDULER.name))
    return lr_scheduler_factory[cfg.SCHEDULER.name](optimizer=optimizer, **params)


def build_lr_scheduler_v2(cfg, optimizer, last_epoch=-1):
    name = cfg.SCHEDULER.name
    config = cfg.SCHEDULER  # retrieve scheduler related config
    step_by = config.step_by
    warm_step = config.warm_step
    warm_step_by = config.warm_step_by
    logging.info("Warm up parameters for scheduler is warm_step={}, warm_step_by={}".format(warm_step, warm_step_by))
    log_info = "Using lr scheduler {}, with parameters: step_by={}, last_epoch={}, ".format(name, step_by, last_epoch)

    if name == 'StepLR':
        log_info += "step_size={}, gamma={}".format(config.step_size, config.step_gamma)
        lr_scheduler = StepLR(
            optimizer, step_size=config.step_size, gamma=config.step_gamma, last_epoch=last_epoch,
            step_by=step_by, warm_step=warm_step, warm_step_by=warm_step_by)
    elif name == 'PolyLR':
        log_info += "max_iter={}, poly_power={}".format(config.max_iter, config.poly_power)
        lr_scheduler = PolyLR(
            optimizer, max_iter=config.max_iter, power=config.poly_power, last_epoch=last_epoch,
            step_by=step_by, warm_step=warm_step, warm_step_by=warm_step_by)
    elif name == 'SquaredLR':
        log_info += "max_iter={}".format(config.max_iter)
        lr_scheduler = SquaredLR(optimizer, max_iter=config.max_iter, last_epoch=last_epoch,
                                 step_by=step_by, warm_step=warm_step, warm_step_by=warm_step_by)
    elif name == 'ExpLR':
        log_info += "exp_step_size={}, exp_gamma={}".format(config.exp_step_size, config.exp_gamma)
        lr_scheduler = ExpLR(
            optimizer, step_size=config.exp_step_size, gamma=config.exp_gamma, last_epoch=last_epoch,
            step_by=step_by, warm_step=warm_step, warm_step_by=warm_step_by)
    else:
        logging.error('Scheduler `{}` not supported'.format(name))
        raise ValueError('Scheduler `{}` not supported'.format(name))
    logging.info(log_info)
    return lr_scheduler
