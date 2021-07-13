import os
import time
import numpy as np
import torch
import torch.nn as nn
from autoattack import AutoAttack

class AAAttack(nn.Module):

	def __init__(self, attack_config):
		super(AAAttack, self).__init__()

	def forward(self, x, y, core_model, loss_fn, mask_fn, eps, mode):
		core_model.eval()

		adversary = AutoAttack(core_model, norm='Linf', eps=eps, version='standard')
		return adversary.run_standard_evaluation(x, y, bs=1000)
