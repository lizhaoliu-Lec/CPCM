import torch
import logging
import torch.nn as nn
import numpy as np
from fvcore.common.registry import Registry

from meta_data.constant import PROJECT_NAME

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for module that creates model for point cloud.

The registered object will be called with `obj(cfg)`.
"""

# from . import unet_3d, relation_net  # noqa F401 isort:skip
from . import resnet, res16unet  # noqa F401 isort:skip

class TwoStreamModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    # the interface should be compatible with regular inference, e.g, only sparse_feats1 is given
    def forward(self, sparse_feats=None, sparse_feats_aux=None, labels=None, labels_aux=None, **kwargs):
        is_training = self.training
        if is_training:

            # for processing single stream masked features only
            single_stream_forward = kwargs.get('single_stream_forward', False)
            if single_stream_forward:
                sparse_feats_single = kwargs.get('sparse_feats_single', None)
                sparse_labels_single = kwargs.get('sparse_labels_single', None)
                assert sparse_feats_single is not None
                assert sparse_labels_single is not None
                return self.model(sparse_feats_single, sparse_labels_single)

            assert sparse_feats_aux is not None, 'sparse_feats_aux not provided during two stream model training'
            assert labels is not None, 'labels not provided during two stream model training'
            assert labels_aux is not None, 'labels_aux not provided during two stream model training'

            # forward the data in stream 1
            ret = self.model(sparse_feats, labels)

            # forward the data in stream 2
            ret_aux = self.model(sparse_feats_aux, labels_aux)

            # combine results from both stream
            for k, v in ret_aux.items():
                ret['{}_aux'.format(k)] = v

            return ret

        else:
            return self.model(sparse_feats)


class MomentumModel(nn.Module):
    def __init__(self, config, model, model_mon):
        super().__init__()
        self.config = config
        self.model = model
        self.model_mon = model_mon
        # momentum model parameters
        self.m = config.MODEL.momentum_model_m

        # assure model parameters are the same
        self.update_momentum_model(global_step=0)

    # the interface should be compatible with regular inference, e.g, only sparse_feats is given
    def forward(self, sparse_feats, sparse_feats_aux=None, labels=None, labels_aux=None, **kwargs):
        is_training = self.training
        if is_training:
            global_step = kwargs.get('global_step', None)

            assert sparse_feats_aux is not None, 'sparse_feats_aux not provided during two stream model training'
            assert labels is not None, 'labels not provided during two stream model training'
            assert labels_aux is not None, 'labels_aux not provided during two stream model training'
            assert global_step is not None, 'global_step not provided during two stream model training'

            # update momentum model
            self.update_momentum_model(global_step)

            # forward the data
            ret = self.model(sparse_feats, labels)

            with torch.no_grad():  # no gradient to momentum model
                ret_aux = self.model_mon(sparse_feats_aux, labels_aux)

            # combine results
            for k, v in ret_aux.items():
                ret['{}_aux'.format(k)] = v

            return ret

        else:
            return self.model(sparse_feats)

    def update_momentum_model(self, global_step):
        # TODO, bug here
        m = min(1 - 1 / (1 + global_step), self.m)
        for param_q, param_k in zip(self.model.parameters(), self.model_mon.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)


class PrototypeModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.num_class = self.config.MODEL.out_channels
        self.prototype_dim = self.config.MODEL.prototype_dim
        self.prototype_update_type = self.config.MODEL.prototype_update_type
        self.entropy_thres = self.config.MODEL.prototype_update_entropy_threshold
        # momentum update prototype
        self.m = config.MODEL.prototype_momentum

        self.register_buffer("prototypes", torch.randn(self.num_class, self.prototype_dim))

    def get_entropy(self, logits):
        # logits: pointwise logits, shape: (N, num_class)
        num_class = torch.tensor(logits.size(1)).float()

        # normalize into (0, 1), shape: (N,)
        return (-1. / torch.log(num_class)) * torch.sum(logits * torch.log(logits), dim=1)

    def update_by_valid_gt(self, feats, labels, global_step):
        # update prototypes based on the feats with labels
        m = min(1 - 1 / (1 + global_step), self.m)

        valid_mask = labels != self.config.DATA.ignore_label

        valid_feats, valid_labels = feats[valid_mask], labels[valid_mask]

        for label in torch.unique(valid_labels):
            class_feats = valid_feats[valid_labels == label, :]

            # average class feats
            class_mean_feat = torch.mean(class_feats, dim=0, keepdim=True)

            self.prototypes.data[label, :] = self.prototypes.data[label, :] * m + class_mean_feat.data * (1 - m)

    def update_by_conf_pred(self, feats, predictions, global_step):
        # update prototypes based on the feats with high confident
        m = min(1 - 1 / (1 + global_step), self.m)

        labels = torch.argmax(predictions, dim=1)  # pseudo label
        entropy = self.get_entropy(torch.softmax(predictions, dim=1))

        valid_mask = entropy < self.entropy_thres

        valid_feats, valid_labels = feats[valid_mask], labels[valid_mask]

        for label in torch.unique(valid_labels):
            class_feats = valid_feats[valid_labels == label, :]

            # average class feats
            class_mean_feat = torch.mean(class_feats, dim=0, keepdim=True)
            self.prototypes.data[label, :] = self.prototypes.data[label, :] * m + class_mean_feat.data * (1 - m)

    def update_prototype(self, feats, labels, predictions, global_step):
        if self.prototype_update_type == 'update_by_valid_gt':
            self.update_by_valid_gt(feats, labels, global_step)
        if self.prototype_update_type == 'update_by_conf_pred':
            self.update_by_conf_pred(feats, predictions, global_step)

    # the interface should be compatible with regular inference, e.g, only sparse_feats is given
    def forward(self, sparse_feats, labels=None, **kwargs):
        is_training = self.training
        if is_training:
            global_step = kwargs.get('global_step', None)
            assert labels is not None, 'labels not provided during two stream model training'
            assert global_step is not None, 'global_step not provided during two stream model training'

            # forward the data
            ret = self.model(sparse_feats, labels)

            # put prototypes to ret
            ret['prototypes'] = self.prototypes.clone()

            with torch.no_grad():  # update prototype latter
                feats = ret['semantic_feats']
                self.update_prototype(feats, labels, predictions=ret['semantic_scores'], global_step=global_step)
            return ret

        else:
            return self.model(sparse_feats)


class LookBackModel(nn.Module):
    def __init__(self, config, model, model_t):
        super().__init__()

        self.config = config
        self.model = model

        self.model_t = model_t
        self.model_t.eval()  # no training for history model

        self.update_every_step = self.config.MODEL.look_back_update_every_step
        self.warm_step = self.config.MODEL.look_back_warm_step

        # assure model parameters are the same
        self.update_history_model(global_step=0)

    def get_entropy(self, logits):
        # logits: pointwise logits, shape: (N, num_class)
        num_class = torch.tensor(logits.size(1)).float()

        # normalize into (0, 1), shape: (N,)
        return (-1. / torch.log(num_class)) * torch.sum(logits * torch.log(logits), dim=1)

    def forward(self, sparse_feats, labels=None, **kwargs):
        is_training = self.training
        if is_training:
            global_step = kwargs.get('global_step', None)

            assert labels is not None, 'labels not provided during look back model training'
            assert global_step is not None, 'global_step not provided during look back training'

            # update momentum model
            self.update_history_model(global_step)

            # forward the data
            ret = self.model(sparse_feats, labels)

            with torch.no_grad():  # no gradient to history model
                # eval for every step, since LookBackModel.train will set all its submodules to training mode
                self.model_t.eval()  # eval or not produces different results
                ret_t = self.model_t(sparse_feats)

                # en1 = self.get_entropy(torch.softmax(ret['semantic_scores'], dim=-1))
                # en2 = self.get_entropy(torch.softmax(ret_t['semantic_scores'], dim=-1))
                #
                # logging.info("==> en1 entropy size: {}, entropy mean: {}".format(en1.size(), torch.mean(en1)))
                # logging.info("==> en2 entropy size: {}, entropy mean: {}".format(en2.size(), torch.mean(en2)))

            # combine results
            for k, v in ret_t.items():
                ret['{}_history'.format(k)] = v

            return ret

        else:
            return self.model(sparse_feats)

    @torch.no_grad()
    def update_history_model(self, global_step):
        is_right_step = global_step < self.warm_step or global_step % self.update_every_step == 0

        if not is_right_step:
            return

        logging.info("Updating history model in global_step={}".format(global_step))
        for param, param_t in zip(self.model.state_dict().values(), self.model_t.state_dict().values()):
            param_t.copy_(param)


def torch_intersect(t1, t2):
    t1 = t1.unique()
    t2 = t2.unique()

    return torch.tensor(np.intersect1d(t1.numpy(), t2.numpy()))


class MLPHeadModel(nn.Module):
    """
    Extra MLP Head for model training
    """

    def __init__(self, config, model):
        super().__init__()

        self.config = config
        self.model = model

        self.expand_factor = self.config.MODEL.mlp_head_expand_factor
        self.dim = self.config.MODEL.mlp_head_dim
        self.instance_categories = torch.tensor(self.get_instance_categories())

        self.mlp_head = self.make_mlp_layer()

    def get_instance_categories(self):
        if 'Scannet' in self.config.DATA.dataset:
            return self.config.DATA.scannet_instance_categories
        if 'Stanford' in self.config.DATA.dataset:
            return self.config.DATA.stanford_instance_categories
        raise ValueError("Unsupported dataset found: {}".format(self.config.DATA.dataset))

    def is_valid_pseudo_instance_label(self, labels):
        is_valid = True

        # filter out the invalid labels
        filtered_labels = labels.clone()

        label_unique, label_count = torch.unique(filtered_labels[filtered_labels != self.config.DATA.ignore_label],
                                                 return_counts=True)

        # filter out not instance category
        label_unique = torch_intersect(label_unique.cpu(), self.instance_categories)

        # only consider instance category

        # three case
        # (0) if no label available
        if label_unique.size(0) == 0:
            is_valid = False
        # (1) if all the same, no_neg
        if label_unique.size(0) == 1:
            is_valid = False
        # (2) if all unique, no_pos
        if label_unique.size(0) == label_count.sum():
            is_valid = False

        if not is_valid:
            logging.warning("Invalid pseudo instance label found, label_unique={}".format(label_unique))

        return is_valid

    def make_mlp_layer(self):
        in_channels = self.dim
        inner_factor = self.expand_factor
        mid_channels = int(in_channels * inner_factor)
        return nn.Sequential(*[
            nn.Linear(in_features=in_channels, out_features=mid_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=in_channels, bias=True),
        ])

    def forward(self, sparse_feats, labels=None, **kwargs):
        is_training = self.training
        if is_training:
            hack_mlp = kwargs.get('hack_mlp', None)
            assert hack_mlp is not None, 'hack_mlp required for MLPHeadModel training'

            # forward the data
            ret = self.model(sparse_feats, labels)

            # get semantic feats
            semantic_feats = ret['semantic_feats']
            # get mlp feats
            mlp_feats = self.mlp_head(semantic_feats)
            ret['mlp_feats'] = mlp_feats

            # check if the label is valid for contrastive loss
            if not self.is_valid_pseudo_instance_label(labels) or hack_mlp:
                # we need to optimize the param in mlp any way, so we do a little hack here
                ret['loss'] += 1e-7 * torch.nn.functional.smooth_l1_loss(
                    mlp_feats, torch.zeros_like(mlp_feats).fill_(torch.mean(mlp_feats))
                )
                # ret['mlp_feats'] = None

            return ret

        else:
            return self.model(sparse_feats)


def build_model(cfg):
    """
    Build a model from `cfg.MODEL.name`.
    """
    logger = logging.getLogger(PROJECT_NAME)
    logger.info('Using model {}'.format(cfg.MODEL.name))
    model = MODEL_REGISTRY.get(cfg.MODEL.name)(cfg)

    if cfg.MODEL.momentum_model_apply:
        # a little hack here
        # deepcopy is a straightforward way, but minkowski does not support
        # therefore, we instantiate a new model, and copy the parameter in the __init__ func of MomentumModel
        model_mon = MODEL_REGISTRY.get(cfg.MODEL.name)(cfg)
        model = MomentumModel(cfg, model, model_mon)

    if cfg.MODEL.two_stream_model_apply:
        model = TwoStreamModel(cfg, model)

    if cfg.MODEL.prototype_model_apply:
        model = PrototypeModel(cfg, model)

    if cfg.MODEL.look_back_model_apply:
        model_t = MODEL_REGISTRY.get(cfg.MODEL.name)(cfg)  # same hack as the momentum model
        model = LookBackModel(cfg, model, model_t)

    if cfg.MODEL.mlp_head_model_apply:
        model = MLPHeadModel(cfg, model)

    return model
