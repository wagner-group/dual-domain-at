'''Code is adapted from
https://github.com/hongyi-zhang/Fixup/blob/master/cifar/models/fixup_resnet_cifar.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ['FixupResNet', 'fixup_resnet20', 'fixup_resnet32',
           'fixup_resnet44', 'fixup_resnet56', 'fixup_resnet110', 'fixup_resnet1202']


class CifNorm(nn.Module):

    def __init__(self):
        super(CifNorm, self).__init__()
        mean = torch.as_tensor((0.4914, 0.4822, 0.4465))[None, :, None, None]
        stdev = torch.as_tensor((0.2023, 0.1994, 0.2010))[None, :, None, None]
        self.register_buffer('mean', mean)
        self.register_buffer('stdev', stdev)

    def forward(self, x):
        return (x - self.mean) / (self.stdev)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

        self.norm_layer1 = nn.Identity()
        self.norm_layer2 = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.norm_layer1(out)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = self.norm_layer2(out)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, core_config, block, layers, num_classes=10):
        super(FixupResNet, self).__init__()
        self.num_classes = core_config['num_classes']
        self.learning_rate = core_config['learning_rate']
        self.lr_decay = core_config['lr_decay']
        self.lr_steps = core_config['lr_steps']
        self.weight_decay = core_config['weight_decay']
        self.normalization_layer = 'none'

        self.first_norm = CifNorm()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes)

        self.norm_layer1 = nn.Identity()

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(
                    m.conv1.weight,
                    mean=0,
                    std=np.sqrt(2 / (m.conv1.weight.shape[0] *
                                     np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

        # Initialize optimizer and LR scheduler
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.lr_steps,
            gamma=self.lr_decay)

    def grad_prep(self):
        self.optimizer.zero_grad()

    def grad_update(self):
        self.optimizer.step()

    def lr_update(self):
        self.lr_scheduler.step()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_norm(x)
        x = self.conv1(x)
        x = self.relu(self.norm_layer1(x) + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x


def fixup_resnet20(core_config, **kwargs):
    """Constructs a Fixup-ResNet-20 model.
    """
    return FixupResNet(core_config, FixupBasicBlock, [3, 3, 3], **kwargs)
