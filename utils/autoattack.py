import torch
from models.wrappers import pair_at

from autoattack import AutoAttack


def run_autoattack(config, wrapper_model, test_loader, device):
    core_model = wrapper_model.module.core_model
    adversary = AutoAttack(
        core_model,
        norm='Linf',
        eps=config['wrapper']['eps_test'],
        version='standard',
    )
    num_total, num_clean, num_adv = 0, 0, 0
    for images, labels in test_loader:
        bs = images.size(0)
        x_adv = adversary.run_standard_evaluation(images, labels, bs=bs)
        num_total += bs
        with torch.no_grad():
            if config['wrapper']['track_stats']:
                y_pred = core_model(torch.cat([images, x_adv], dim=0).to(device)).argmax(1).cpu()
                wrapper_model.module.stats_tracker()
                num_clean += (y_pred[:bs] == labels).sum().item()
                num_adv += (y_pred[bs:] == labels).sum().item()
            else:
                num_adv += (core_model(x_adv.to(device)).argmax(1).cpu() == labels).sum().item()
                num_clean += (core_model(images.to(device)).argmax(1).cpu() == labels).sum().item()
        if num_total >= config['test']['num_sample']:
            break
    return num_adv / num_total, num_clean / num_total


def update_stats(act_all, act, batch_idx):
    for key in act:
        mean, var = act[key].mean([0, 2, 3]), act[key].var([0, 2, 3], unbiased=False)
        if batch_idx == 0:
            act_all[key] = {'mean': mean, 'var': var}
        else:
            act_all[key]['mean'] = (act_all[key]['mean'] * batch_idx + mean) / (batch_idx + 1)
            act_all[key]['var'] = (act_all[key]['var'] * batch_idx + var) / (batch_idx + 1)
    return act_all


def finetune_bn(config, wrapper_model, test_loader, device):
    core_model = wrapper_model.module.core_model
    adversary = AutoAttack(
        core_model,
        norm='Linf',
        eps=config['wrapper']['eps_test'],
        version='standard',
    )
    num_total = 0
    adv_act, cl_act = {}, {}
    for batch_idx, (images, labels) in enumerate(test_loader):
        x_adv = adversary.run_standard_evaluation(images, labels, bs=images.size(0))
        num_total += images.size(0)
        with torch.no_grad():
            core_model(x_adv.to(device))
            update_stats(adv_act, wrapper_model.module.activations[str(device)], batch_idx)
            core_model(images.to(device))
            update_stats(cl_act, wrapper_model.module.activations[str(device)], batch_idx)
        if num_total >= config['test']['num_sample']:
            break

    print('Setting new mean/var...')
    for name, layer in core_model.named_modules():
        if not name in adv_act:
            continue

        if isinstance(wrapper_model.module, pair_at.PairATWrapper):
            layer.running_mean = (adv_act[name]['mean'] + cl_act[name]['mean']) / 2
            layer.running_var = (adv_act[name]['var'] + cl_act[name]['var']) / 2
            continue

        if layer.mode == 'clean':
            layer.bn1.running_mean = (adv_act[name]['mean'] + cl_act[name]['mean']) / 2
            layer.bn1.running_var = (adv_act[name]['var'] + cl_act[name]['var']) / 2
        elif layer.mode == 'gmm':
            layer.gmm_mean1 = cl_act[name]['mean']
            layer.gmm_mean2 = adv_act[name]['mean']
            layer.gmm_var1 = cl_act[name]['var']
            layer.gmm_var2 = adv_act[name]['var']
        else:
            raise NotImplementedError()
