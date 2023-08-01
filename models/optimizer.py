from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer


class SNRAdam(Optimizer):
    r"""Implements the SNRAdam optimization algorithm, which uses std deviation for the denominator rather than
    sqrt(energy) term used in conventional Adam. Why is this a good idea? If gradient stddev for a param is small, we
    should take larger steps as it means the gradient is consistent over time.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its variance (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            weight_decay: float = 0.0,
            eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError(
                        'SNRAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                if len(state) == 0:
                    state['iter_'] = 1
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                iter_ = state['iter_']
                exp_avg = state['exp_avg']
                if iter_ == 1:
                    d_sub_p_sq = d_p - exp_avg
                else:
                    d_sub_p_sq = d_p - exp_avg.mul(1.0 / (1 - beta1 ** (iter_ - 1)))
                d_sub_p_sq.mul_(d_sub_p_sq)

                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(d_p, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(d_sub_p_sq, alpha=1.0 - beta2)

                p.data.addcdiv_(exp_avg.mul(1.0 / (1 - beta1 ** iter_)),
                                exp_avg_sq.mul(1.0 / (1 - beta2 ** iter_)).sqrt() + eps, value=-lr)
                state['iter_'] += 1

        return loss
