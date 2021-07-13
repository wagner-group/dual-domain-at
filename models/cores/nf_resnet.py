import os
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class NF_ResNet_Core(nn.Module):

    def __init__(self, core_config):
        super(NF_ResNet_Core, self).__init__()

        # Load core hyperparameters
        self.num_classes = core_config['num_classes']
        self.learning_rate = core_config['learning_rate']
        self.lr_decay = core_config['lr_decay']
        self.lr_steps = core_config['lr_steps']
        self.weight_decay = core_config['weight_decay']
        self.classifier = timm.create_model(
            'nf_resnet26', pretrained=False, num_classes=self.num_classes)
        self.clip_grad = core_config.get('clip_grad', None)
        self.use_asc = core_config.get('use_asc', False)
        self.normalization_layer = 'none'

        self.first_norm = CifNorm()
        self.init_opt(-1)

    def init_opt(self, last_epoch):
        # Initialize optimizer and LR scheduler
        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay)
        if self.use_asc:
            print('Using adaptive gradient clipping...')
            self.optimizer = AGC(self.classifier.parameters(), self.optimizer, clipping=0.16)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.lr_steps,
            gamma=self.lr_decay)
        # Dummy loop to get lr_scheduler to continue on after the reset
        for _ in range(last_epoch + 1):
            self.lr_scheduler.step()

    def grad_prep(self):
        self.optimizer.zero_grad()

    def grad_update(self):
        if self.clip_grad is not None and not self.use_nfnet:
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.clip_grad)
        # print(self.layer1[0].conv1.weight.grad.norm())
        # print(self.layer2[0].conv1.weight.grad.norm())
        # print(self.layer3[0].conv1.weight.grad.norm())
        # print(self.layer4[0].conv1.weight.grad.norm())
        self.optimizer.step()

    def lr_update(self):
        self.lr_scheduler.step()

    def forward(self, x):
        x = self.first_norm(x)
        return self.classifier(x)
