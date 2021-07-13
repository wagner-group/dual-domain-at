import torch.nn as nn


def init_bn(num_features, **kwargs):
    return nn.BatchNorm2d(num_features)


def init_gn(num_features, num_group=32, **kwargs):
    return nn.GroupNorm(num_group, num_features)


def init_in(num_features, **kwargs):
    return nn.InstanceNorm2d(num_features, track_running_stats=False)


def init_none(num_features, **kwargs):
    return nn.Identity()
