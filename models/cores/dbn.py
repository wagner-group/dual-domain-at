import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBatchNorm2d(nn.Module):
    eps = 1e-5

    def __init__(self, num_features, momentum=0.1, mode='clean',
                 shared_affine=True, temperature=1., score_func=None,
                 score_norm_func='sigmoid', combine_mode=None, **kwargs):
        super(DualBatchNorm2d, self).__init__()
        self.forward_func = {
            'clean': self.forward_clean,
            'adv': self.forward_adv,
            'sep': self.forward_sep,
            'soft': self.forward_soft,
            'hard': self.forward_hard,
            'gmm': self.forward_gmm,
            'igmm': self.forward_igmm,
        }
        self.mode = mode
        self.shared_affine = shared_affine
        self.momentum = momentum
        self.temperature = temperature
        self.score_func = score_func
        self.score_norm_func = score_norm_func
        self.combine_mode = combine_mode
        self.reg = {}

        self.bn1 = nn.BatchNorm2d(
            num_features,
            affine=not shared_affine,
            momentum=momentum,
        )
        self.bn2 = nn.BatchNorm2d(
            num_features,
            affine=not shared_affine,
            momentum=momentum,
        )

        self.register_buffer('gmm_mean1', torch.zeros(num_features))
        self.register_buffer('gmm_mean2', torch.zeros(num_features))
        self.register_buffer('gmm_var1', torch.ones(num_features))
        self.register_buffer('gmm_var2', torch.ones(num_features))

        if shared_affine:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.weight1 = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.weight2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.bias1 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
            self.bias2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    def normalize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def scale(self, x, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return x * weight[None, :, None, None] + bias[None, :, None, None]

    @staticmethod
    def gaussian_kld(mu1, mu2, var1, var2):
        return 0.5 * torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5

    @staticmethod
    def gaussian_wd(mu1, mu2, var1, var2):
        return (mu1 - mu2) ** 2 + var1 + var2 - 2 * (var1 * var2).sqrt()

    def log_prob(self, x, mean, var):
        mean = mean[None, :, None, None]
        var = var[None, :, None, None] + self.eps
        log_prob = (2 * math.pi * var).log() + (x - mean) ** 2 / var
        return -0.5 * log_prob

    def prob_gmm(self, x, mu1, mu2, var1, var2):
        ns, _, h, w = x.size()
        log_prob1 = self.log_prob(x, mu1, var1)
        log_prob2 = self.log_prob(x, mu2, var2)
        score_cl, score_adv = log_prob1.sum(1), log_prob2.sum(1)
        logits = score_cl - score_adv
        score = torch.sigmoid(logits)
        if self.training:
            labels = torch.zeros(ns * h * w, device=x.device, requires_grad=False)
            labels[:ns * h * w // 2] = 1
            self.reg[str(x.device)] = F.binary_cross_entropy_with_logits(logits.view(-1), labels)
        return score

    def update_stats(self, mean, var, n, run_mean=None, run_var=None):
        run_mean *= (1 - self.momentum)
        run_mean += self.momentum * mean
        run_var *= (1 - self.momentum)
        run_var += self.momentum * var * n / (n - 1)

    def update_running_stats(self, bn, x, mean=None, var=None):
        # Update separate running stats for clean and adv
        if mean is None:
            mean = x.mean([0, 2, 3])
        if var is None:
            var = x.var([0, 2, 3], unbiased=False)
        n = x.numel() / x.size(1) / 2
        with torch.no_grad():
            args = {
                '1': {'run_mean': self.bn1.running_mean, 'run_var': self.bn1.running_var},
                '2': {'run_mean': self.bn2.running_mean, 'run_var': self.bn2.running_var},
                'gmm1': {'run_mean': self.gmm_mean1, 'run_var': self.gmm_var1},
                'gmm2': {'run_mean': self.gmm_mean2, 'run_var': self.gmm_var2},
            }[bn]
            self.update_stats(mean, var, n, **args)

    def get_prob_score(self, x):
        return self.prob_gmm(x, self.bn1.running_mean, self.bn2.running_mean,
                             self.bn1.running_var, self.bn2.running_var)

    def get_kld_score(self, x):
        mean, var = x.mean([2, 3]), x.var([2, 3], unbiased=False) + self.eps
        score_cl = self.gaussian_kld(self.bn1.running_mean[None, :], mean,
                                     self.bn1.running_var[None, :], var).mean(1)
        score_adv = self.gaussian_kld(self.bn2.running_mean[None, :], mean,
                                      self.bn2.running_var[None, :], var).mean(1)
        score = 1 / (1 + ((-score_adv + score_cl) / self.temperature).clamp(-50, 50).exp())
        return score

    def compute_score(self, x):
        size = x.size(0) // 2
        score = {
            'prob': self.get_prob_score,
            'kld': self.get_kld_score,
        }[self.score_func](x)
        self.reg[str(x.device)] = -(score[:size].log() + (1 - score[size:]).log()).mean() / 2
        return score

    def combine_norm(self, p, x1, x2):
        if self.training:
            if self.combine_mode == 'thres':
                p = (p >= 0.5).float()
            elif self.combine_mode == 'fixed':
                p = torch.zeros_like(p)
                p[:p.size(0) // 2] = 1
        return p * x1 + (1 - p) * x2

    def forward_gmm(self, x):

        size = x.size(0) // 2
        if self.training:
            mu1, var1 = x[:size].mean([0, 2, 3]), x[:size].var([0, 2, 3], unbiased=False)
            mu2, var2 = x[size:].mean([0, 2, 3]), x[size:].var([0, 2, 3], unbiased=False)
            self.update_running_stats('1', x[:size], mu1, var1)
            self.update_running_stats('2', x[size:], mu2, var2)
        else:
            mu1, var1 = self.bn1.running_mean, self.bn1.running_var
            mu2, var2 = self.bn2.running_mean, self.bn2.running_var

        x_norm1 = self.normalize(x, mu1[None, :, None, None], var1[None, :, None, None])
        x_norm2 = self.normalize(x, mu2[None, :, None, None], var2[None, :, None, None])
        p_cl = self.prob_gmm(x, mu1, mu2, var1, var2)

        p_cl = p_cl[:, None, :, :]
        if self.shared_affine:
            x_norm = self.combine_norm(p_cl, x_norm1, x_norm2)
            x_norm = self.scale(x_norm)
        else:
            x_norm1 = self.scale(x_norm1, weight=self.weight1, bias=self.bias1)
            x_norm2 = self.scale(x_norm2, weight=self.weight2, bias=self.bias2)
            x_norm = self.combine_norm(p_cl, x_norm1, x_norm2)

        return x_norm

    def forward_clean(self, x):
        out = self.bn1(x)
        if self.shared_affine:
            out = self.scale(out)
        return out

    def forward_adv(self, x):
        out = self.bn2(x)
        if self.shared_affine:
            out = self.scale(out)
        return out

    def forward_sep(self, x):
        size = x.size(0) // 2
        out_cl = self.bn1(x[:size])
        out_adv = self.bn2(x[size:])
        out = torch.cat([out_cl, out_adv], dim=0)
        if self.shared_affine:
            out = self.scale(out)
        return out

    def forward_hard(self, x):
        weight = self.compute_score(x)[:, None]
        mask = (weight > 0.5).float()
        mean = mask * self.bn1.running_mean + (1 - mask) * self.bn2.running_mean
        var = mask * self.bn1.running_var + (1 - mask) * self.bn2.running_var
        out = self.normalize(x, mean, var)
        if self.shared_affine:
            out = self.scale(out)
        else:
            raise NotImplementedError()
        if self.training:
            size = x.size(0) // 2
            self.update_running_stats('1', x[:size])
            self.update_running_stats('2', x[size:])
        return out

    def forward(self, x):
        out = self.forward_func[self.mode](x)
        assert ~out.isnan().any()
        return out
