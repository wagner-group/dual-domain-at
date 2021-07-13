import os
import time
import numpy as np
import torch
import torch.nn as nn
from attacks import kl_pgd


class TRADESWrapper(nn.Module):

    def __init__(self, core_model, eval_attacker, config):
        super(TRADESWrapper, self).__init__()
        self.training_epoch = 1
        self.core_model = core_model
        self.eval_attacker = eval_attacker
        self.tr_attacker = kl_pgd.KL_PGD_Attack(
            config['attack']).to(eval_attacker.device)

        # Load wrapper hyperparameters
        wrapper_config = config['wrapper']
        self.eps_train = wrapper_config['eps_train']
        self.eps_test = wrapper_config['eps_test']
        self.adv_beta = wrapper_config['adv_beta']
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(1)
        self.log_softmax = nn.LogSoftmax(1)
        self.concat = wrapper_config['trades_concat']
        self.latest_reg = 0

    def loss_fn(self, logits, y, r='mean'):
        return nn.CrossEntropyLoss(reduction=r)(logits, y)

    def kl_loss(self, cl_logits, adv_logits):
        cl_probs = self.softmax(cl_logits)
        adv_lprobs = self.log_softmax(adv_logits)
        return self.kl_criterion(adv_lprobs, cl_probs)

    def train_wf(self, x, y):
        eps = self.eps_train

        # Get adversarial logits from attacker
        x_adv = self.tr_attacker(x, y, self.core_model, self.loss_fn, None, eps, True)
        if self.concat:
            logits = self.core_model(torch.cat((x, x_adv), 0))
            bs = x.size(0)
            cl_logits, adv_logits = logits[:bs], logits[bs:]
        else:
            cl_logits = self.core_model(x)
            adv_logits = self.core_model(x_adv)

        # Compute/return loss from adversarial inputs
        adv_loss = self.kl_loss(cl_logits, adv_logits)
        return self.loss_fn(cl_logits, y) + self.adv_beta * adv_loss

    def eval_wf(self, x, y):
        eps = self.eps_test

        # Get adversarial logits from attacker
        cl_logits, adv_logits = self.eval_attacker(
            x, y, self.core_model, self.loss_fn, None, eps, False)

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
