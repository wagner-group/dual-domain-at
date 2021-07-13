import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.cores.bn_util import init_bn, init_gn, init_in, init_none
from models.cores.dbn import DualBatchNorm2d
from nfnets import AGC


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, **kwargs):
        super(PreActBlock, self).__init__()

        self.bn1 = norm_layer(in_planes, **kwargs)
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, **kwargs)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(
                in_planes, self.expansion * planes,
                kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(
            out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


# class PreActBottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes,
#                                planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes,
#                                self.expansion*planes, kernel_size=1, bias=False)
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(nn.Conv2d(
#                 in_planes, self.expansion*planes,
#                 kernel_size=1, stride=stride, bias=False))

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(
#             out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         return out + shortcut


class CifNorm(nn.Module):

    def __init__(self):
        super(CifNorm, self).__init__()
        mean = torch.as_tensor((0.4914, 0.4822, 0.4465))[None, :, None, None]
        stdev = torch.as_tensor((0.2023, 0.1994, 0.2010))[None, :, None, None]
        self.register_buffer('mean', mean)
        self.register_buffer('stdev', stdev)

    def forward(self, x):
        return (x - self.mean) / (self.stdev)


class PreActResNet_20_Core(nn.Module):

    def __init__(self, core_config):
        super(PreActResNet_20_Core, self).__init__()

        # Load core hyperparameters
        self.num_classes = core_config['num_classes']
        self.learning_rate = core_config['learning_rate']
        self.lr_decay = core_config['lr_decay']
        self.lr_steps = core_config['lr_steps']
        self.weight_decay = core_config['weight_decay']
        self.clip_grad = core_config.get('clip_grad', None)
        self.use_nfnet = core_config.get('use_nfnet', False)

        # Define feature extractor/classifier
        self.in_planes = 64
        self.first_norm = CifNorm()
        self.normalization_layer = core_config['normalization_layer']
        self.norm_layer = {
            'bn': init_bn,
            'gn': init_gn,
            'in': init_in,
            'none': init_none,
            'dbn': DualBatchNorm2d,
        }[core_config['normalization_layer']]
        block = PreActBlock
        self.expand = block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(block, 64, 2, 1, **core_config)
        self.layer2 = self.make_layer(block, 128, 2, 2, **core_config)
        self.layer3 = self.make_layer(block, 256, 2, 2, **core_config)
        self.layer4 = self.make_layer(block, 512, 2, 2, **core_config)
        self.linear = nn.Linear(512 * self.expand, self.num_classes)
        self.init_opt(-1)

    def init_opt(self, last_epoch):
        # Initialize optimizer and LR scheduler
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay)
        if self.use_nfnet:
            print('Using adaptive gradient clipping (NFNet)...')
            self.optimizer = AGC(self.parameters(), self.optimizer, clipping=0.16)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.lr_steps,
            gamma=self.lr_decay)
        # Dummy loop to get lr_scheduler to continue on after the reset
        for _ in range(last_epoch + 1):
            self.lr_scheduler.step()

    def make_layer(self, block, planes, num_blocks, stride, **kwargs):
        layers, strides = [], [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                norm_layer=self.norm_layer, **kwargs))
            self.in_planes = planes * self.expand
        return nn.Sequential(*layers)

    def set_dbn_mode(self, mode):
        i = 0
        for name, layer in self.named_modules():
            if isinstance(layer, DualBatchNorm2d):
                if isinstance(mode, list):
                    layer.mode = mode[i]
                else:
                    layer.mode = mode
                i += 1

    def set_dbn_eval(self, mode):
        for name, layer in self.named_modules():
            if isinstance(layer, DualBatchNorm2d):
                if layer.mode == mode:
                    layer.training = False

    def grad_prep(self):
        self.optimizer.zero_grad()

    def grad_update(self):
        if self.clip_grad is not None and not self.use_nfnet:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
        self.optimizer.step()

    def lr_update(self):
        self.lr_scheduler.step()

    def forward(self, x):
        if x.isnan().any():
            raise ValueError('NaN detected!')
        x = self.first_norm(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        return self.linear(x.view(x.size(0), -1))
