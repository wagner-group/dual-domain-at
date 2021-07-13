import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from attacks import aa, noatt, pgd
from models.cores import fixup_resnet, nf_resnet, prn_20, resnet
from models.cores.dbn import DualBatchNorm2d
from models.wrappers import at, clean, match_at, pair_at, trades
from utils import autoattack, data, fio, log, mdl

parser = argparse.ArgumentParser(description='Test config file')
parser.add_argument('config_file', type=str, default='config/config.yml',
                    help='path to config file')
args = parser.parse_args()
config_file = args.config_file

# Load meta hyperparameters
config = fio.yaml_load(config_file)
exp_id = config['meta']['exp_id']
dataset = config['meta']['dataset']
exp_core = config['meta']['exp_core']
exp_wrapper = config['meta']['exp_wrapper']
exp_attacker = config['meta']['exp_attacker']
data_path = config['meta']['data_path']
results_path = config['meta']['results_path']
gpu_ids = config['meta']['gpu_ids']
random_seed = config['meta']['random_seed']

# Load training hyperparameters
batch_size = config['test']['batch_size']
num_workers = config['train']['num_workers']

# Load testing hyperparameters
exp_name = config['test']['exp_name']
load_epoch = config['test']['load_epoch']

# Set random seeds
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Obtain directory with results
exp_res_path = os.path.join(results_path, dataset, exp_wrapper, exp_core)
if not exp_name:
    exp_name = sorted(os.listdir(exp_res_path))[-1]
exp_res_path = os.path.join(exp_res_path, exp_name)
log_save_path = os.path.join(exp_res_path, 'tests.txt')
model_save_path = os.path.join(exp_res_path, 'checkpoints')

# Initialize logger
lg = log.Logger(log_save_path, config)
lg.print('Logger initialized')

# Initialize GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lg.print(f'Using {device} for testing')

# Get test data
data_loader = {
    'mnist': data.get_mnist,
    'cifar10': data.get_cifar10,
    'cifar100': data.get_cifar100,
    'imagenette': data.get_imagenette,
}[dataset]
_, _, test_loader = data_loader(
    data_path, batch_size, num_workers, 0, random_seed)

# Define core model
core_builder = {
    'prn_20': prn_20.PreActResNet_20_Core,
    'frn_20': fixup_resnet.fixup_resnet20,
    'nf_resnet_26': nf_resnet.NF_ResNet_Core,
    'resnet34': resnet.resnet34,
}[exp_core]
core_model = core_builder(config['core']).to(device)

# Define eval attacker
attack_builder = {
    'noatt': noatt.NoAttack,
    'pgd': pgd.PGDAttack,
    'aa': aa.AAAttack,
}[exp_attacker]
eval_attacker = attack_builder(config['attack']).to(device)

# Define wrapper model
wrapper_builder = {
    'clean': clean.CleanWrapper,
    'at': at.ATWrapper,
    'trades': trades.TRADESWrapper,
    'pair_at': pair_at.PairATWrapper,
    'match_at': match_at.MatchATWrapper,
}[exp_wrapper]
eval_attacker.device = device
wrapper_model = wrapper_builder(core_model, eval_attacker, config)
if device == 'cuda':
    wrapper_model = nn.DataParallel(wrapper_model)
    torch.backends.cudnn.benchmark = True

# Load desired epoch checkpoint into core model
epoch_state = torch.load(os.path.join(
    model_save_path, f'epoch_{load_epoch:02d}.pth'))
core_model.load_state_dict(epoch_state)
core_model.eval()

if config['meta']['exp_wrapper'] == 'match_at':
    mode = wrapper_model.module.attack_mode
    core_model.set_dbn_mode(mode)
    lg.print('Mode in use...')
    for name, layer in core_model.named_modules():
        if isinstance(layer, DualBatchNorm2d):
            lg.print(layer.mode)

if config['test']['use_aa']:
    adv_acc, clean_acc = autoattack.run_autoattack(config, wrapper_model, test_loader, device)
    lg.print(f'Clean: {clean_acc * 100:.2f} | Adv: {adv_acc * 100:.2f}')

if config['test']['use_pgd']:
    # Print description of job
    lg.print('-------------------------------------------')
    lg.print(f'Testing {exp_name} with:')
    lg.print(f'Dataset: {dataset}')
    lg.print(f'Core model: {exp_core}')
    lg.print(f'Wrapper method: {exp_wrapper}')
    lg.print(f'Evaluation attacker: {exp_attacker}')
    lg.print(f'Model checkpoint: epoch_{load_epoch:02d}.pth')
    lg.print('-------------------------------------------')

    # Evaluate loaded model on the test set
    mdl.evaluate(config, wrapper_model, test_loader, device, lg, 'test')
    lg.write('-------------------------------------------------------')

if config['wrapper']['track_stats']:
    pickle.dump(wrapper_model.module.stats, open(f'stats_{exp_name}.pkl', 'wb'))
