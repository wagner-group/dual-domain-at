import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks import kl_pgd, pgd


class PairATWrapper(nn.Module):
    eps = 1e-5

    def __init__(self, core_model, eval_attacker, config):
        super(PairATWrapper, self).__init__()
        self.training_epoch = 1
        self.core_model = core_model
        self.is_dp = isinstance(core_model, nn.DataParallel)
        self.eval_attacker = eval_attacker

        # Load wrapper hyperparameters
        wrapper_config = config['wrapper']
        self.eps_train = wrapper_config['eps_train']
        self.eps_test = wrapper_config['eps_test']
        self.dist_reg = wrapper_config['dist_reg']
        self.reg_weight = wrapper_config['reg_weight']
        self.concat = wrapper_config['concat']
        self.loss_weight = wrapper_config.get('loss_weight', None)
        if self.loss_weight is not None:
            assert 0 <= self.loss_weight <= 1
        self.normalize = wrapper_config['normalize_dim']
        self.pair_samples = wrapper_config['pair_samples']

        self.use_trades = wrapper_config.get('pair_trades', False)
        self.use_kld = wrapper_config.get('pair_kld', False)
        self.adv_beta = wrapper_config['pair_adv_beta']
        if self.use_trades:
            self.tr_attacker = kl_pgd.KL_PGD_Attack(config['attack']).to(eval_attacker.device)
        else:
            self.tr_attacker = pgd.PGDAttack(config['attack']).to(eval_attacker.device)

        # Register hook to get BN output
        self.latest_reg = 0
        self.activations = {}
        self.norm_layer = {
            'bn': nn.BatchNorm2d,
            'gn': nn.GroupNorm,
            'in': nn.InstanceNorm2d,
            'none': nn.Identity,
        }[self.core_model.normalization_layer]
        self.num_bn_layer = 0
        self.idx_to_name = []
        for name, layer in self.core_model.named_modules():
            if isinstance(layer, self.norm_layer):
                layer.register_forward_hook(self.setup_hook(
                    name, wrapper_config['pair_reg_use_input']))
                self.idx_to_name.append(name)
                self.num_bn_layer += 1

        self.track_stats = wrapper_config.get('track_stats', False)
        self.last_tracked = 0
        self.stats = {
            'mean_clean': [[] for _ in range(self.num_bn_layer)],
            'var_clean': [[] for _ in range(self.num_bn_layer)],
            'mean_adv': [[] for _ in range(self.num_bn_layer)],
            'var_adv': [[] for _ in range(self.num_bn_layer)],
            'running_mean': [[] for _ in range(self.num_bn_layer)],
            'running_var': [[] for _ in range(self.num_bn_layer)],
            'samples': [[] for _ in range(self.num_bn_layer)],
        }

    def setup_hook(self, name, use_input=True):
        def hook(model, input, output):
            key = str(input[0].device)
            if key not in self.activations:
                self.activations[key] = {}
            if use_input:
                self.activations[key][name] = input[0] if isinstance(input, tuple) else input
            else:
                self.activations[key][name] = output[0] if isinstance(output, tuple) else output
        return hook

    @staticmethod
    def kl_loss(cl_logits, adv_logits):
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        return F.kl_div(adv_lprobs, cl_probs, reduction='batchmean')

    @staticmethod
    def gaussian_kld(mu1, mu2, sigma1, sigma2):
        return 0.5 * torch.log(sigma2 / sigma1) + (sigma1 + (mu1 - mu2) ** 2) / (2 * sigma2) - 0.5

    @staticmethod
    def gaussian_wd(mu1, mu2, sigma1, sigma2):
        wd = (mu1 - mu2) ** 2 + sigma1 + sigma2 - 2 * (sigma1 * sigma2).sqrt()
        return wd

    def get_running_stats(self):
        stats = []
        for _, layer in self.core_model.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                stats.append([layer.running_mean.cpu().numpy(), layer.running_var.cpu().numpy()])
        return stats

    def compute_reg(self, in_clean, in_adv, running_stats=None):
        if self.pair_samples:
            dim = [2, 3]
        else:
            dim = [0, 2, 3]
        mu_clean, mu_adv = in_clean.mean(dim), in_adv.mean(dim)
        var_clean, var_adv = in_clean.var(dim), in_adv.var(dim)
        var_clean.clamp_min_(self.eps)
        var_adv.clamp_min_(self.eps)

        if self.dist_reg == 'kld_clean':
            reg_layer = self.gaussian_kld(mu_clean, mu_adv, var_clean, var_adv)
        elif self.dist_reg == 'kld_adv':
            reg_layer = self.gaussian_kld(mu_adv, mu_clean, var_adv, var_clean)
        elif self.dist_reg == 'l2':
            assert self.pair_samples
            reg_layer = ((in_clean - in_adv) ** 2)
        elif self.dist_reg == 'wd':
            reg_layer = self.gaussian_wd(mu_clean, mu_adv, var_clean, var_adv)
        else:
            raise NotImplementedError('dist_reg not implemented!')

        if self.pair_samples:
            reg_layer = reg_layer.mean(0)

        if self.normalize:
            return reg_layer.mean()
        return reg_layer.sum()

    def regularizer(self, device, act_clean=None, act_adv=None):
        reg = 0
        if self.num_bn_layer == 0 or self.dist_reg == 'none':
            return reg

        if act_clean is not None:
            num_layers = len(act_clean)
            assert num_layers == len(act_adv)
        else:
            acts = list(self.activations[str(device)].values())

        if self.dist_reg == 'kld_run':
            stats = self.get_running_stats()
        else:
            stats = None

        for i in range(self.num_bn_layer):
            if act_clean is not None:
                in_clean, in_adv = act_clean[i], act_adv[i]
            else:
                activation = acts[i]
                size = activation.size(0) // 2
                in_clean, in_adv = activation[:size], activation[size:]
            reg += self.compute_reg(in_clean, in_adv, running_stats=stats)
        self.latest_reg = reg.item()
        return reg

    def stats_tracker(self, clean=True):
        with torch.no_grad():
            stats = self.get_running_stats()
            for i in range(self.num_bn_layer):
                if len(stats) > 0:
                    self.stats['running_mean'][i].append(stats[i][0])
                    self.stats['running_var'][i].append(stats[i][1])
                activation = []
                for device in self.activations.keys():
                    activation.append(self.activations[device][self.idx_to_name[i]])
                activation = torch.cat(activation, dim=0)
                dim = [0, 2, 3]
                if self.concat:
                    size = activation.size(0) // 2
                    in_clean, in_adv = activation[:size], activation[size:]
                else:
                    in_clean = activation if clean else None
                    in_adv = activation if not clean else None
                if in_clean is not None:
                    mu_clean, var_clean = in_clean.mean(dim).cpu().numpy(), in_clean.var(dim).cpu().numpy()
                    self.stats['mean_clean'][i].append(mu_clean)
                    self.stats['var_clean'][i].append(var_clean)
                if in_adv is not None:
                    mu_adv, var_adv = in_adv.mean(dim).cpu().numpy(), in_adv.var(dim).cpu().numpy()
                    self.stats['mean_adv'][i].append(mu_adv)
                    self.stats['var_adv'][i].append(var_adv)
                self.stats['samples'][i].append(activation.cpu().numpy())

    def loss_fn(self, logits, y, r='mean'):
        return nn.CrossEntropyLoss(reduction=r)(logits, y)

    def get_activations(self, device):
        activations = []
        for act in self.activations[str(device)].values():
            activations.append(act.clone())
        return activations

    def train_wf(self, x, y):
        eps = self.eps_train
        bs = x.size(0)

        track = False
        self.last_tracked += 1
        if self.track_stats and self.last_tracked == 50:
            self.last_tracked = 0
            track = True

        # Get adversarial logits from attacker
        x_adv = self.tr_attacker(x, y, self.core_model, self.loss_fn, None, eps, True)
        if self.concat:
            logits = self.core_model(torch.cat((x, x_adv), 0))
            logits_clean, logits_adv = logits[:bs], logits[bs:]
            reg = self.regularizer(x.device)
            if track:
                self.stats_tracker()
        else:
            logits_clean = self.core_model(x)
            if track:
                self.stats_tracker(True)
            act_clean = self.get_activations(x.device)
            logits_adv = self.core_model(x_adv)
            if track:
                self.stats_tracker(False)
            act_adv = self.get_activations(x.device)
            logits = torch.cat((logits_clean, logits_adv), 0)
            reg = self.regularizer(x.device, act_clean, act_adv)

        if self.use_kld:
            adv_loss = self.kl_loss(logits_clean, logits_adv)
            loss_clean = self.loss_fn(logits_clean, y)
            loss_adv = self.loss_fn(logits_adv, y)
            return self.loss_weight * loss_clean + (1 - self.loss_weight) * loss_adv + \
                self.adv_beta * adv_loss + reg

        if self.use_trades:
            adv_loss = self.kl_loss(logits_clean, logits_adv)
            return self.loss_fn(logits_clean, y) + self.adv_beta * adv_loss + self.reg_weight * reg

        if self.loss_weight is None:
            return self.loss_fn(logits, torch.cat((y, y), 0)) + self.reg_weight * reg

        loss_clean = self.loss_fn(logits[:bs], y)
        loss_adv = self.loss_fn(logits[bs:], y)
        return self.loss_weight * loss_clean + (1 - self.loss_weight) * loss_adv + self.reg_weight * reg

    def eval_wf(self, x, y):
        eps = self.eps_test
        bs = x.size(0)

        # Get adversarial logits from attacker
        x_adv = self.eval_attacker(x, y, self.core_model, self.loss_fn, None, eps, False)
        logits = self.core_model(torch.cat((x, x_adv), 0))
        cl_logits, adv_logits = logits[:bs], logits[bs:]

        if self.track_stats:
            self.stats_tracker()

        # Compute/return loss from adversarial inputs
        cl_loss_sum = self.loss_fn(cl_logits, y, r='sum')
        adv_loss_sum = self.loss_fn(adv_logits, y, r='sum')
        return cl_logits, adv_logits, cl_loss_sum, adv_loss_sum

    def update_step(self):
        return

    def update_epoch(self, lg):
        self.training_epoch += 1

        # Update core model LR
        if self.is_dp:
            self.core_model.module.lr_update()
        else:
            self.core_model.lr_update()

    def forward(self, x, y):
        return self.train_wf(x, y) if self.training else self.eval_wf(x, y)
