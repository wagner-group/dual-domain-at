import torch
import torch.nn as nn


class PGDAttack(nn.Module):

    def __init__(self, attack_config):
        super(PGDAttack, self).__init__()

        # Load attack hyperparameters
        self.pgd_steps = attack_config['pgd_steps']
        self.pgd_step_size = attack_config['pgd_step_size']
        self.num_restarts = attack_config['num_restarts']

    def forward(self, x, y, core_model, loss_fn, mask_fn, eps, mode):
        core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1).to(x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.pgd_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = core_model(x_adv)
                    loss = loss_fn(logits, y)
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Compute gradient update mask
                    mask = torch.ones_like(x_adv)
                    if mask_fn:
                        mask = mask_fn(logits)

                    # Perform gradient update, project to norm ball
                    x_adv = x_adv.detach() + (
                        self.pgd_step_size * mask * torch.sign(grads))
                    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)

                    # Clip perturbed inputs to image domain
                    x_adv = torch.clamp(x_adv, 0, 1)

            # Update worst-case inputs with itemized final losses
            fin_losses = loss_fn(core_model(x_adv), y, r='none').reshape(worst_losses.shape)
            up_mask = (fin_losses >= worst_losses).float()
            x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
            worst_losses = fin_losses * up_mask + worst_losses * (1 - up_mask)

        # Return worst-case perturbed input logits
        core_model.train(mode)
        return x_adv_worst.detach()
