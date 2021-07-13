import argparse
import logging
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from attacks import aa, noatt, pgd
from models.cores import fixup_resnet, nf_resnet, prn_20, resnet
from models.wrappers import at, clean, match_at, pair_at, trades
from utils import autoattack, data, fio, log, mdl

parser = argparse.ArgumentParser(description='Train config file')
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
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']
val_ratio = config['train']['val_ratio']
train_epochs = config['train']['train_epochs']

# Set random seeds
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Create directory to push results
exp_name = f'e{exp_id}_{fio.get_exp_hash()}'
exp_res_path = os.path.join(results_path, dataset, exp_wrapper, exp_core, exp_name)
log_save_path = os.path.join(exp_res_path, 'log.txt')
model_save_path = os.path.join(exp_res_path, 'checkpoints')
config_save_path = os.path.join(exp_res_path, 'tr_config.yml')
os.makedirs(model_save_path)
os.system(f'cp {config_file} {config_save_path}')

# Initialize logger
lg = log.Logger(log_save_path, config)
fio.setup_text_logger('tmp', log_dir='/tmp/', append=False, log_level=logging.DEBUG, console_out=True)
lg.print('Logger initialized')

# Initialize GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lg.print(f'Using {device} for training')

# Get train/val/test data
data_loader = {
    'mnist': data.get_mnist,
    'cifar10': data.get_cifar10,
    'cifar100': data.get_cifar100,
    'imagenette': data.get_imagenette,
}[dataset]
train_loader, val_loader, test_loader = data_loader(
    data_path, batch_size, num_workers, val_ratio, random_seed)

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

# Print description of job
lg.print('-------------------------------------------')
lg.print(f'Training {exp_name} with:')
lg.print(f'Dataset: {dataset}')
lg.print(f'Core model: {exp_core}')
lg.print(f'Wrapper method: {exp_wrapper}')
lg.print(f'Evaluation attacker: {exp_attacker}')
lg.print('-------------------------------------------')

if config['train']['load_pretrain']:
    lg.print('Loading checkpoint...')
    pt_exp_name = config['test']['exp_name']
    pt_exp_res_path = os.path.join(results_path, dataset, exp_wrapper, exp_core, pt_exp_name)
    pt_model_save_path = os.path.join(pt_exp_res_path, 'checkpoints')
    load_epoch = config['test']['load_epoch']
    epoch_state = torch.load(os.path.join(pt_model_save_path, f'epoch_{load_epoch:02d}.pth'))
    core_model.load_state_dict(epoch_state)
    wrapper_model.module.update_epoch(lg)

# Train model for specified number of epochs
best_acc, best_epoch = 0, 0
for e in range(1, train_epochs + 1):
    lg.print(f'Training epoch {e} of {train_epochs}')

    # Backpropagate on each batch
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # Get model output
        wrapper_model.train()
        core_model.grad_prep()
        train_loss = wrapper_model(x, y)

        # Backpropagate the loss
        train_loss.mean().backward()
        core_model.grad_update()
        wrapper_model.module.update_step()

        if train_loss.isnan().any():
            raise ValueError('NaN loss!')

        # Print statistics
        if batch_idx % 100 == 0:
            message = f'Batch: {batch_idx}, loss: {train_loss.mean().item():.4f}'
            if isinstance(wrapper_model.module, (pair_at.PairATWrapper, match_at.MatchATWrapper)):
                message += f', reg: {wrapper_model.module.latest_reg:.4f}'
            lg.print(message)

    # Periodically save and evaluate model
    if e % 2 == 0:
        # Save model checkpoint
        torch.save(core_model.state_dict(), os.path.join(model_save_path, f'epoch_{e:02d}.pth'))
        lg.print(f'Saved checkpoint epoch_{e:02d}.pth')

        # Evaluate model on the validation set
        with torch.no_grad():
            adv_acc = mdl.evaluate(config, wrapper_model, val_loader, device, lg, 'val')
        if adv_acc >= best_acc:
            best_acc = adv_acc
            best_epoch = e

    # Update epoch-level parameters
    wrapper_model.module.update_epoch(lg)

    # Reset best acc if there's fine-tuning
    if isinstance(wrapper_model.module, match_at.MatchATWrapper):
        if config['wrapper']['match_tune_epoch'] == e:
            best_acc = 0

if config['wrapper']['track_stats']:
    pickle.dump(wrapper_model.module.stats, open(f'train_stats_{exp_name}.pkl', 'wb'))

# Load best epoch
path = os.path.join(model_save_path, f'epoch_{best_epoch:02d}.pth')
core_model.load_state_dict(torch.load(path))
core_model.eval()
# Evaluate fully-trained model on the test set
mdl.evaluate(wrapper_model, test_loader, device, lg, 'test')

lg.print(f'Best adv acc: {best_acc * 100:.2f}')
_, _, test_loader = data_loader(
    data_path, config['test']['batch_size'], num_workers, val_ratio, random_seed)
adv_acc, clean_acc = autoattack.run_autoattack(config, wrapper_model, test_loader, device)
lg.print(f'Clean: {clean_acc * 100:.2f} | Adv: {adv_acc * 100:.2f}')
