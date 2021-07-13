import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KL_PGD_Attack(nn.Module):

    def __init__(self, attack_config):
        super(KL_PGD_Attack, self).__init__()

        # Load attack hyperparameters
        self.pgd_steps = attack_config['pgd_steps']
        self.pgd_step_size = attack_config['pgd_step_size']
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def kl_loss(self, cl_logits, adv_logits):
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        return self.kl_criterion(adv_lprobs, cl_probs)

    def forward(self, x, y, core_model, loss_fn, mask_fn, eps, mode):
        core_model.eval()

        # Initialize adversarial inputs
        x_cl = x.clone().detach()
        x_adv = x.clone().detach()
        x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)

        # Run KL-PGD on inputs for specified number of steps
        for _ in range(self.pgd_steps):
            x_adv = x_adv.requires_grad_()

            # Compute logits, loss, gradients
            cl_logits = core_model(x_cl)
            adv_logits = core_model(x_adv)
            loss = self.kl_loss(cl_logits, adv_logits)
            grads = torch.autograd.grad(loss, x_adv)[0].detach()

            # Perform gradient update, project to norm ball
            x_adv = x_adv.detach() + (
                self.pgd_step_size * torch.sign(grads))
            x_adv = torch.min(torch.max(x_adv, x_cl - eps), x_cl + eps)

            # Clip perturbed inputs to image domain
            x_adv = torch.clamp(x_adv, 0, 1)

        # Return worst-case perturbed input logits
        core_model.train(mode)
        return x_adv
