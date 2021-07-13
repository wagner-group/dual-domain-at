import os
import time
import numpy as np
import torch
import torch.nn as nn


class CleanWrapper(nn.Module):

    def __init__(self, core_model, eval_attacker, config):
        super(CleanWrapper, self).__init__()
        self.training_epoch = 1
        self.core_model = core_model
        self.eval_attacker = eval_attacker

        # Load wrapper hyperparameters
        wrapper_config = config['wrapper']
        self.eps_train = wrapper_config['eps_train']
        self.eps_test = wrapper_config['eps_test']

    def loss_fn(self, logits, y, r='mean'):
        return nn.CrossEntropyLoss(reduction=r)(logits, y)

    def train_wf(self, x, y):
        eps = self.eps_train

        # No attack if training
        self.core_model.train()
        cl_logits = self.core_model(x)
        return self.loss_fn(cl_logits, y)

    def eval_wf(self, x, y):
        eps = self.eps_test
        size = x.size(0)

        # Get adversarial logits from attacker
        x_adv = self.eval_attacker(x, y, self.core_model, self.loss_fn, None, eps, False)
        logits = self.core_model(torch.cat((x, x_adv), 0))
        cl_logits, adv_logits = logits[:size], logits[size:]

        # Compute/return loss from adversarial inputs
        cl_loss_sum = self.loss_fn(cl_logits, y, r='sum')
        adv_loss_sum = self.loss_fn(adv_logits, y, r='sum')
        return cl_logits, adv_logits, cl_loss_sum, adv_loss_sum

    def update_step(self):
        return

    def update_epoch(self, lg):
        self.training_epoch += 1

        # Update core model LR
        self.core_model.lr_update()

    def forward(self, x, y):
        return self.train_wf(x, y) if self.training else self.eval_wf(x, y)
