'''
This code is taken from
https://github.com/yaodongyu/TRADES/blob/master/models/wideresnet.py
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.cores.bn_util import init_bn, init_gn, init_in, init_none
from models.cores.dbn import DualBatchNorm2d
from nfnets import AGC


class CifNorm(nn.Module):

    def __init__(self):
        super(CifNorm, self).__init__()
        mean = torch.as_tensor((0.4914, 0.4822, 0.4465))[None, :, None, None]
        stdev = torch.as_tensor((0.2023, 0.1994, 0.2010))[None, :, None, None]
        self.register_buffer('mean', mean)
        self.register_buffer('stdev', stdev)

    def forward(self, x):
        return (x - self.mean) / (self.stdev)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, norm_layer=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.bn1 = norm_layer(in_planes, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(out_planes, **kwargs)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 dropRate=0.0, norm_layer=None, **kwargs):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate, norm_layer, **kwargs)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate, norm_layer, **kwargs):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, dropRate,
                                norm_layer=norm_layer, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, core_config, depth=28, widen_factor=10, dropRate=0.0, **kwargs):
        super(WideResNet, self).__init__()

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

        nChannels = [16, 16 * widen_factor,
                     32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate,
                                   norm_layer=self.norm_layer, **core_config)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate,
                                       norm_layer=self.norm_layer, **core_config)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate,
                                   norm_layer=self.norm_layer, **core_config)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate,
                                   norm_layer=self.norm_layer, **core_config)
        # global average pooling and classifier
        self.bn1 = self.norm_layer(nChannels[3], **core_config)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], self.num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

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
        # print(self.layer1[0].conv1.weight.grad.norm())
        # print(self.layer2[0].conv1.weight.grad.norm())
        # print(self.layer3[0].conv1.weight.grad.norm())
        # print(self.layer4[0].conv1.weight.grad.norm())
        self.optimizer.step()

    def lr_update(self):
        self.lr_scheduler.step()

    def forward(self, x):
        x = self.first_norm(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def wideresnet28_10(core_config):
    return WideResNet(core_config, depth=28, widen_factor=10, dropRate=0.)
