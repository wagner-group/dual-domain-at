import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks import kl_pgd, pgd
from models.cores.dbn import DualBatchNorm2d


class MatchATWrapper(nn.Module):
    eps = 1e-5

    def __init__(self, core_model, eval_attacker, config):
        super(MatchATWrapper, self).__init__()
        self.training_epoch = 1
        self.core_model = core_model
        self.is_dp = isinstance(core_model, nn.DataParallel)
        self.eval_attacker = eval_attacker

        # Load wrapper hyperparameters
        wrapper_config = config['wrapper']
        self.eps_train = wrapper_config['eps_train']
        self.eps_test = wrapper_config['eps_test']
        self.concat = wrapper_config['match_concat']
        self.loss_weight = wrapper_config.get('match_loss_weight', None)
        if self.loss_weight is not None:
            assert 0 <= self.loss_weight <= 1
        self.reg_func = wrapper_config['match_reg_func']
        self.reg_weight = wrapper_config['match_reg_weight']
        self.latest_reg = 0

        self.train_mode = wrapper_config['match_train_mode']
        print('Train mode: ', self.train_mode)
        self.eval_mode = wrapper_config.get('match_eval_mode', self.train_mode)
        print('Eval mode: ', self.eval_mode)
        self.attack_mode = wrapper_config.get('match_attack_mode', None)
        if self.attack_mode is None:
            self.attack_mode = ['adv' if m == 'sep' else m for m in self.train_mode]
        print('Attack mode: ', self.attack_mode)
        self.tune_mode = wrapper_config.get('match_tune_mode', None)
        if self.tune_mode is None:
            self.tune_mode = ['soft' if m == 'sep' else m for m in self.train_mode]
        self.tune_epoch = wrapper_config.get('match_tune_epoch', 9999)
        self.tune_fix_bn = wrapper_config['match_tune_fix_bn']
        self.tune_fix_dbn = wrapper_config['match_tune_fix_dbn']
        self.tuning = False

        self.use_trades = wrapper_config.get('match_trades', False)
        self.use_kld = wrapper_config.get('match_kld', False)
        self.adv_beta = wrapper_config['match_adv_beta']
        if self.use_trades:
            self.tr_attacker = kl_pgd.KL_PGD_Attack(config['attack']).to(eval_attacker.device)
        else:
            self.tr_attacker = pgd.PGDAttack(config['attack']).to(eval_attacker.device)

        self.track_stats = wrapper_config.get('track_stats', False)
        self.last_tracked = 0
        self.num_bn_layer = 0
        self.activations = {}
        self.idx_to_name = []
        for name, layer in self.core_model.named_modules():
            if isinstance(layer, DualBatchNorm2d):
                layer.register_forward_hook(self.setup_hook(
                    name, wrapper_config['match_reg_use_input']))
                self.idx_to_name.append(name)
                self.num_bn_layer += 1
        print(f'Num normalization layers: {self.num_bn_layer}')
        # TODO:
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

    def get_dbn_mode(self, mode):
        idx = []
        i = 0
        for name, layer in self.core_model.named_modules():
            if isinstance(layer, DualBatchNorm2d):
                if layer.mode == mode:
                    idx.append(i)
                i += 1
        return idx

    def get_running_stats(self):
        stats = []
        for _, layer in self.core_model.named_modules():
            if isinstance(layer, DualBatchNorm2d):
                if layer.mode == 'gmm':
                    stats.append([[layer.gmm_mean1.cpu().numpy(), layer.gmm_mean2.cpu().numpy()],
                                  [layer.gmm_var1.cpu().numpy(), layer.gmm_var2.cpu().numpy()]])
                else:
                    stats.append([[layer.bn1.running_mean.cpu().numpy(), layer.bn2.running_mean.cpu().numpy()],
                                  [layer.bn1.running_var.cpu().numpy(), layer.bn2.running_var.cpu().numpy()]])
        return stats

    def stats_tracker(self):
        assert self.concat
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
                size = activation.size(0) // 2
                in_clean, in_adv = activation[:size], activation[size:]
                dim = [0, 2, 3]
                mu_clean, mu_adv = in_clean.mean(dim).cpu().numpy(), in_adv.mean(dim).cpu().numpy()
                var_clean, var_adv = in_clean.var(dim).cpu().numpy(), in_adv.var(dim).cpu().numpy()
                self.stats['mean_clean'][i].append(mu_clean)
                self.stats['mean_adv'][i].append(mu_adv)
                self.stats['var_clean'][i].append(var_clean)
                self.stats['var_adv'][i].append(var_adv)
                self.stats['samples'][i].append(activation.cpu().numpy())

    @staticmethod
    def kl_loss(cl_logits, adv_logits):
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        return F.kl_div(adv_lprobs, cl_probs, reduction='batchmean')

    @staticmethod
    def gaussian_wd(mu1, mu2, sigma1, sigma2):
        return (mu1 - mu2) ** 2 + sigma1 + sigma2 - 2 * (sigma1 * sigma2).sqrt()

    @staticmethod
    def gaussian_kld(mu1, mu2, sigma1, sigma2):
        return 0.5 * torch.log(sigma2 / sigma1) + (sigma1 + (mu1 - mu2) ** 2) / (2 * sigma2) - 0.5

    def dbn_score(self, device):
        # Assume concat batch (first half is clean and second is adv)
        reg = 0
        for _, layer in self.core_model.named_modules():
            if isinstance(layer, DualBatchNorm2d):
                # TODO: adapt to other func and handle clean mode
                if not str(device) in layer.reg:
                    continue
                if isinstance(layer.reg[str(device)], torch.Tensor):
                    reg += layer.reg[str(device)]
        return reg

    def compute_reg(self, in_clean, in_adv):
        dim = [0, 2, 3]
        mu_clean, mu_adv = in_clean.mean(dim), in_adv.mean(dim)
        var_clean, var_adv = in_clean.var(dim), in_adv.var(dim)
        var_clean.clamp_min_(self.eps)
        var_adv.clamp_min_(self.eps)
        if var_clean.isnan().any():
            raise ValueError('NaN detected!')
        if self.reg_func == 'wd_mean':
            reg_layer = - self.gaussian_wd(mu_clean, mu_adv, var_clean, var_adv).mean()
        elif self.reg_func == 'wd_sum':
            reg_layer = - self.gaussian_wd(mu_clean, mu_adv, var_clean, var_adv).sum()
        elif self.reg_func == 'kld_clean_mean':
            reg_layer = - self.gaussian_kld(mu_clean, mu_adv, var_clean, var_adv).mean()
        elif self.reg_func == 'kld_clean_sum':
            reg_layer = - self.gaussian_kld(mu_clean, mu_adv, var_clean, var_adv).sum()
        elif 'neg_kld_clean' in self.reg_func:
            reg_layer = self.gaussian_kld(mu_clean, mu_adv, var_clean, var_adv).sum()
        elif 'neg_wd' in self.reg_func:
            reg_layer = self.gaussian_wd(mu_clean, mu_adv, var_clean, var_adv).mean()

        return reg_layer

    def regularizer(self, device):
        if self.reg_func == 'none':
            return 0
        if 'score' in self.reg_func:
            reg = self.dbn_score(device)
        else:
            reg = 0

        if not 'score' in self.reg_func or '-' in self.reg_func:
            acts = self.activations[str(device)]
            reg_idx = self.get_dbn_mode('gmm')
            reg_idx.extend(self.get_dbn_mode('igmm'))
            for i in range(self.num_bn_layer):
                if i not in reg_idx:
                    continue
                activation = acts[self.idx_to_name[i]]
                size = activation.size(0) // 2
                in_clean, in_adv = activation[:size], activation[size:]
                reg += self.compute_reg(in_clean, in_adv) * (1e+2 if '-' in self.reg_func else 1)

        if reg != 0:
            self.latest_reg = reg.item()
        return reg

    def loss_fn(self, logits, y, r='mean'):
        return nn.CrossEntropyLoss(reduction=r)(logits, y)

    def train_wf(self, x, y):
        eps = self.eps_train
        bs = x.size(0)

        # This mode will be used during the attack
        self.core_model.set_dbn_mode(self.attack_mode)
        x_adv = self.tr_attacker(x, y, self.core_model, self.loss_fn, None, eps, True)

        if self.tuning and self.tune_fix_bn:
            self.core_model.set_dbn_eval('clean')
        if self.tuning and self.tune_fix_dbn:
            self.core_model.set_dbn_eval('soft')
            self.core_model.set_dbn_eval('gmm')

        # This mode will be used for training
        self.core_model.set_dbn_mode(self.train_mode)
        if self.concat:
            logits = self.core_model(torch.cat((x, x_adv), 0))
            logits_clean, logits_adv = logits[:bs], logits[bs:]
        else:
            logits_clean = self.core_model(x)
            logits_adv = self.core_model(x_adv)
            logits = torch.cat((logits_clean, logits_adv), 0)
        reg_ = self.regularizer(x.device)
        reg = self.reg_weight * reg_

        if self.track_stats:
            self.last_tracked += 1
            if self.last_tracked == 10:
                self.last_tracked = 0
                self.stats_tracker()

        if self.use_kld:
            adv_loss = self.kl_loss(logits_clean, logits_adv)
            loss_clean = self.loss_fn(logits_clean, y)
            loss_adv = self.loss_fn(logits_adv, y)
            return self.loss_weight * loss_clean + (1 - self.loss_weight) * loss_adv + \
                self.adv_beta * adv_loss + reg

        if self.use_trades:
            adv_loss = self.kl_loss(logits_clean, logits_adv)
            return self.loss_fn(logits_clean, y) + self.adv_beta * adv_loss + reg

        if self.loss_weight is None:
            return self.loss_fn(logits, torch.cat((y, y), 0)) + reg

        loss_clean = self.loss_fn(logits[:bs], y)
        loss_adv = self.loss_fn(logits[bs:], y)
        return self.loss_weight * loss_clean + (1 - self.loss_weight) * loss_adv + reg

    def eval_wf(self, x, y):

        eps = self.eps_test
        bs = x.size(0)

        self.core_model.set_dbn_mode(self.attack_mode)
        x_adv = self.eval_attacker(x, y, self.core_model, self.loss_fn, None, eps, False)

        self.core_model.set_dbn_mode(self.eval_mode)
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
        if self.training_epoch == self.tune_epoch:
            print('Start fine-tuning...')
            self.tuning = True
            self.train_mode = self.tune_mode
            self.eval_mode = self.tune_mode
            self.attack_mode = self.tune_mode
            print(self.train_mode, self.eval_mode, self.attack_mode)
            print('Re-initialize optimizer...')
            self.core_model.init_opt(self.training_epoch - 1)

        # Update core model LR
        if self.is_dp:
            self.core_model.module.lr_update()
        else:
            self.core_model.lr_update()

        self.training_epoch += 1

    def forward(self, x, y):
        return self.train_wf(x, y) if self.training else self.eval_wf(x, y)
