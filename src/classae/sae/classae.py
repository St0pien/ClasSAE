import json
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lapsum.topk import soft_topk

from classae.sae.config import SAEConfig
from classae.sae.core import (
    Dictionary,
    SAETrainer,
    geometric_median,
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)


class ClasSAE(Dictionary, nn.Module):
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        num_classes: int,
        rho: int,
        alpha: float,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_classes = num_classes
        self.rho = rho

        self.register_buffer("k", torch.tensor(float(k), dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        self.class_matrix = nn.Parameter(torch.randn(dict_size, num_classes) * 0.01)
        self.budget_vector = nn.Parameter(
            torch.zeros((dict_size,), dtype=torch.float32)
        )

    def calculate_M(self):
        Ktot = float(self.rho * self.dict_size)

        # Compute per feature association budget
        k = Ktot * torch.softmax(self.budget_vector, dim=0)
        k = torch.clamp(k, 1e-8, float(self.num_classes))

        M = soft_topk(self.class_matrix, k.unsqueeze(-1), self.alpha.clone(), dim=1)

        return M

    def encode(
        self, x: torch.Tensor, return_active: bool = False, use_threshold: bool = True
    ):
        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts = post_relu_feat_acts * (post_relu_feat_acts > self.threshold)
        else:
            flattened_acts = post_relu_feat_acts.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts = (
                torch.zeros_like(post_relu_feat_acts.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts.shape)
            )

        if return_active:
            return (
                encoded_acts,
                encoded_acts.sum(0) > 0,
                post_relu_feat_acts,
            )
        else:
            return encoded_acts

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = True):
        encoded_acts = self.encode(x)
        x_hat = self.decode(encoded_acts)

        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "ClasSAE":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(
            f"{path}/ae.pt",
            map_location=device,
            weights_only=True,
        )

        # Handle checkpoints saved from DataParallel / DDP.
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[len("module.") :]: value for key, value in state_dict.items()
            }

        # Infer dimensions from the state dict.
        decoder_weight = state_dict["decoder.weight"]
        activation_dim, dict_size = decoder_weight.shape

        class_matrix = state_dict["class_matrix"]
        class_matrix_dict_size, num_classes = class_matrix.shape

        if class_matrix_dict_size != dict_size:
            raise ValueError(
                f"Inconsistent dict_size: decoder.weight has {dict_size}, "
                f"but class_matrix has {class_matrix_dict_size}"
            )

        # alpha is a registered scalar buffer, so it is recoverable.
        alpha = state_dict["alpha"].item()

        with open(f"{path}/config.json") as f_config:
            json_config = json.load(f_config)
            rho = json_config["sae"]["rho"]

        if "k" in state_dict:
            k_init = state_dict["k"].item()
        else:
            k_init = json_config["sae"].get("k")
            if k_init is None:
                raise ValueError(
                    f"Checkpoint at {path} has no 'k' buffer (state_dict) and no "
                    "'sae.k' field in config.json. This checkpoint predates the "
                    "k-debiasing fix; specify the intended target k explicitly "
                    "rather than let this silently default to something wrong."
                )

        model = cls(
            activation_dim=activation_dim,
            dict_size=dict_size,
            num_classes=num_classes,
            rho=rho,
            alpha=alpha,
            k=k_init,
        )

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        return model


@dataclass
class ClasSAEConfig(SAEConfig):
    k: int = 64
    soft_topk_alpha: float = 0.001
    alpha_anneal_steps: Optional[int] = None
    num_classes: int = 1000
    rho: int = 5
    agreement_loss_weight: float = 1.0
    agreement_tau: float = 1.0
    tau_anneal_start: float = 50.0
    tau_anneal_steps: Optional[int] = None


class ClasSAETrainer(SAETrainer):
    ae: ClasSAE

    def __init__(self, steps, cfg: ClasSAEConfig):
        super().__init__(steps, cfg)
        self.k = cfg.k
        self.steps = steps
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps
        self.k_anneal_steps = cfg.k_anneal_steps
        self.threshold_beta = cfg.threshold_beta
        self.threshold_start_step = cfg.threshold_start_step
        self.agreement_loss_weight = cfg.agreement_loss_weight
        self.soft_topk_alpha = cfg.soft_topk_alpha
        self.alpha_anneal_steps = cfg.alpha_anneal_steps

        self.agreement_tau = cfg.agreement_tau
        self.tau_anneal_start = cfg.tau_anneal_start
        self.tau_anneal_steps = cfg.tau_anneal_steps
        self.threshold_beta = cfg.threshold_beta

        self.active_tau = cfg.agreement_tau

        self.ae = ClasSAE(
            cfg.activation_dim,
            cfg.dict_size,
            cfg.k,
            cfg.num_classes,
            cfg.rho,
            cfg.soft_topk_alpha,
        )

        if cfg.lr is not None:
            self.lr = cfg.lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = cfg.dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = cfg.auxk_alpha
        self.dead_feature_threshold = cfg.dead_feature_threshold
        self.topk_aux = cfg.activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = torch.zeros(cfg.dict_size, dtype=torch.long)

        ### LOGGING SETUP
        self.logging_parameters = [
            "dead_features",
            "pre_norm_auxk_loss",
            "avg_k",
            "min_k",
            "max_k",
            "agreement_loss",
            "ae_soft_topk_alpha",
            "active_tau",
            "use_hard_topk",
            "lr_log",
            "avg_enc_grad",
            "track_threshold",
        ]
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.avg_k = -1
        self.min_k = -1
        self.max_k = -1
        self.agreement_loss = -1
        self.ae_soft_topk_alpha = 1
        self.use_hard_topk = 0
        self.avg_enc_grad = 0
        self.avg_mlp_grad = 0
        self.track_threshold = self.ae.threshold

        ### LOGGING SETUP

        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, cfg.warmup_steps, decay_start=cfg.decay_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )
        self.lr_log = self.scheduler.get_last_lr()[0]

    def to(self, *args, **kwargs):
        self.ae.to(*args, **kwargs)
        self.num_tokens_since_fired = self.num_tokens_since_fired.to(*args, **kwargs)

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None or k_anneal_steps == 0:
            return

        assert (
            0 <= k_anneal_steps < self.steps
        ), "k_anneal_steps must be >= 0 and < steps."
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        self.ae.k.fill_(int(round(annealed_value)))

    def update_annealed_alpha(
        self, step: int, alpha_anneal_steps: Optional[int] = None
    ):
        if alpha_anneal_steps is None or alpha_anneal_steps == 0:
            return

        assert (
            0 <= alpha_anneal_steps < self.steps
        ), "alpha_anneal_steps must be >= 0 and < steps."

        step = min(step, alpha_anneal_steps)
        ratio = step / alpha_anneal_steps
        annealed_value = (1 - ratio) + self.soft_topk_alpha * ratio
        self.ae.alpha.fill_(annealed_value)

    def update_annealed_tau(self, step: int, tau_anneal_steps: Optional[int] = None):
        if tau_anneal_steps is None or tau_anneal_steps == 0:
            return

        assert (
            0 <= tau_anneal_steps < self.steps
        ), "tau_anneal_steps must be >= 0 and < steps."

        step = min(step, tau_anneal_steps)
        ratio = step / tau_anneal_steps
        annealed_value = (
            self.tau_anneal_start * (1 - ratio) + self.agreement_tau * ratio
        )
        self.active_tau = annealed_value

    def update_threshold(self, f: torch.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = torch.tensor(0.0, dtype=torch.float)
            else:
                min_activation = active.min().detach().to(dtype=torch.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.topk_aux, dead_features.sum())

            auxk_latents = torch.where(
                dead_features[None], post_relu_acts_BF, -torch.inf
            )

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def get_agreement_loss(
        self, p: torch.Tensor, k_hat: torch.Tensor, labels: torch.Tensor
    ):
        # Normalize feature_selection mass
        pi = p / k_hat.unsqueeze(-1).detach()  # [B, d]

        # Get M matrix
        M = self.ae.calculate_M()  # [d, C]

        # Scores against every class, not just the true one
        s = pi @ M  # [B, C]

        tau = self.active_tau

        # True-class score
        s_true = s.gather(1, labels.unsqueeze(-1)).squeeze(-1)  # [B]

        # Contrastive form: -s_cb/tau + logsumexp(s/tau)
        log_denom = torch.logsumexp(s / tau, dim=1)  # [B]

        loss = -s_true / tau + log_denom

        return loss.mean()

    def loss(self, x, y, step=None, logging=False):
        (
            f_hard,
            active_indices_F,
            post_relu_acts,
        ) = self.ae.encode(x, return_active=True, use_threshold=False)

        if step > self.threshold_start_step and self.ae.training:
            self.update_threshold(f_hard)

        k_hat = (f_hard > 0).sum(dim=-1, dtype=torch.int64)
        weights = soft_topk(
            post_relu_acts, k_hat.float().unsqueeze(1), self.ae.alpha.clone()
        )
        f_soft = post_relu_acts * weights

        f_combined = f_hard + (f_soft - f_soft.detach())

        x_hat = self.ae.decode(f_combined)

        e = x - x_hat

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        self.avg_k = k_hat.mean(dtype=torch.float32)
        self.min_k = k_hat.min()
        self.max_k = k_hat.max()
        self.ae_soft_topk_alpha = self.ae.alpha.clone()
        self.lr_log = self.scheduler.get_last_lr()[0]
        self.track_threshold = self.ae.threshold

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts)

        agreement_loss = self.get_agreement_loss(weights, k_hat, y)
        self.agreement_loss = agreement_loss

        loss = (
            l2_loss
            + self.auxk_alpha * auxk_loss
            + self.agreement_loss_weight * agreement_loss
        )

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f_hard,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x, y):
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        loss = self.loss(x, y, step=step)
        loss.backward()

        self.avg_enc_grad = (
            self.ae.encoder.weight.grad.mean().item()
            if self.ae.encoder.weight.grad is not None
            else 0
        )

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)
        self.update_annealed_alpha(step, self.alpha_anneal_steps)
        self.update_annealed_tau(step, self.tau_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()
