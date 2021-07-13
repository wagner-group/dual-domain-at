import os
import time
import numpy as np
import torch
import torch.nn as nn

class NoAttack(nn.Module):

	def __init__(self, attack_config):
		super(NoAttack, self).__init__()

	def forward(self, x, y, core_model, loss_fn, mask_fn, eps, mode):
		core_model.train(mode)
		cl_logits = None if mode else core_model(x)
		return cl_logits, core_model(x.clone().detach())
