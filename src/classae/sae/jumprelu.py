from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.autograd as autograd
import torch.nn as nn

from classae.sae.config import SAEConfig

from .core import (
    Dictionary,
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    """Straight-through estimator for the jump-ReLU gate, used only in the
    trainer's loss computation so gradients flow into `threshold`. The SAE's
    own `encode` (used at inference / plain forward passes) uses a hard,
    non-differentiable gate instead — see JumpReluSAE.encode."""

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReluSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(dict_size, activation_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.threshold = nn.Parameter(torch.ones(dict_size) * 0.001)  # Appendix I

        self.apply_b_dec_to_input = False

        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(
        self,
        x: torch.Tensor,
        output_pre_jump: bool = False,
        return_active: bool = False,
    ):
        x = x.to(self.W_enc.dtype)
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        # Hard gate, no straight-through estimator here — matches the
        # original JumpReluAutoEncoder.encode exactly. The differentiable
        # version (JumpReLUFunction) is used only inside the trainer's loss.
        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))

        if return_active:
            return f, f.sum(0) > 0
        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "JumpReluSAE":
        state_dict = torch.load(f"{path}/ae.pt")
        activation_dim, dict_size = state_dict["W_enc"].shape

        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


@dataclass
class JumpReluSAEConfig(SAEConfig):
    bandwidth: float = 0.001
    sparsity_penalty: float = 1.0
    sparsity_warmup_steps: Optional[int] = 2000
    target_l0: float = 20.0


class JumpReluTrainer(SAETrainer):
    """
    Trains a JumpReLU autoencoder (https://arxiv.org/abs/2407.14435).

    Note: unlike the top-k family of trainers, this has no auxiliary loss
    for dead features or k-annealing — sparsity is enforced entirely via
    the target_l0 penalty, and gradients into `threshold` flow through the
    JumpReLUFunction/StepFunction straight-through estimators computed
    directly in `loss()` (NOT through self.ae.encode(), which uses a hard,
    non-differentiable gate).
    """

    def __init__(self, steps: int, cfg: JumpReluSAEConfig):
        super().__init__(steps, cfg)
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps
        self.bandwidth = cfg.bandwidth
        self.sparsity_coefficient = cfg.sparsity_penalty
        self.sparsity_warmup_steps = cfg.sparsity_warmup_steps
        self.target_l0 = cfg.target_l0

        self.ae = JumpReluSAE(cfg.activation_dim, cfg.dict_size)

        # Paper default; unlike the top-k trainers this does NOT use the
        # 1/sqrt(dict_size) auto-LR heuristic even if cfg.lr is None.
        self.lr = cfg.lr if cfg.lr is not None else 7e-5

        # Parameters from the paper: note betas=(0.0, 0.999), not (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.0, 0.999), eps=1e-8
        )

        lr_fn = get_lr_schedule(
            steps,
            cfg.warmup_steps,
            cfg.decay_start,
            resample_steps=None,
            sparsity_warmup_steps=cfg.sparsity_warmup_steps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            steps, cfg.sparsity_warmup_steps
        )

        # Purely for logging purposes — no auxiliary loss here, unlike the
        # top-k trainers' num_tokens_since_fired.
        self.dead_feature_threshold = cfg.dead_feature_threshold
        self.num_tokens_since_fired = torch.zeros(cfg.dict_size, dtype=torch.long)
        self.dead_features = -1
        self.logging_parameters = ["dead_features"]

    def to(self, *args, **kwargs):
        self.ae.to(*args, **kwargs)
        self.num_tokens_since_fired = self.num_tokens_since_fired.to(*args, **kwargs)

    def loss(self, x, step=None, logging=False):
        # Note: using threshold directly, not log_threshold — simpler and
        # avoids complicating scale_biases(). This recomputes pre_jump/f via
        # the differentiable JumpReLUFunction rather than calling
        # self.ae.encode(), since the latter's hard gate has no gradient
        # w.r.t. threshold.
        sparsity_scale = self.sparsity_warmup_fn(step)
        x = x.to(self.ae.W_enc.dtype)

        pre_jump = x @ self.ae.W_enc + self.ae.b_enc
        f = JumpReLUFunction.apply(pre_jump, self.ae.threshold, self.bandwidth)

        active_indices = f.sum(0) > 0
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += x.size(0)
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (
            (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()
        )

        x_hat = self.ae.decode(f)

        l2_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()

        sparsity_loss = (
            self.sparsity_coefficient
            * ((l0 / self.target_l0) - 1).pow(2)
            * sparsity_scale
        )
        loss = l2_loss + sparsity_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x, _):
        loss = self.loss(x, step=step)
        loss.backward()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T,
            self.ae.W_dec.grad.T,
            self.ae.activation_dim,
            self.ae.dict_size,
        ).T
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()
