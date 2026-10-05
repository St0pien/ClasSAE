from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from classae.sae.config import SAEConfig

from .core import (
    ConstrainedAdam,
    Dictionary,
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
)


class StandardSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        # initialize encoder and decoder weights
        w = torch.randn(activation_dim, dict_size)
        # normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        # set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x: torch.Tensor, return_active: bool = False):
        f = nn.functional.relu(self.encoder(x - self.bias))
        if return_active:
            return f, f.sum(0) > 0
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False, ghost_mask=None):
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.functional.relu(f_pre)

            # note that this only applies the decoder weight matrix, no bias
            x_ghost = self.decoder(f_ghost)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.bias.data *= scale

    def normalize_decoder(self):
        norms = torch.norm(self.decoder.weight, dim=0).to(
            dtype=self.decoder.weight.dtype, device=self.decoder.weight.device
        )

        if torch.allclose(norms, torch.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = torch.randn(10, self.activation_dim)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = torch.norm(self.decoder.weight, dim=0)
        assert torch.allclose(new_norms, torch.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert torch.allclose(initial_output, new_output, atol=1e-4)

    @classmethod
    def from_pretrained(
        cls, path, dtype=torch.float, device=None, normalize_decoder=True, **kwargs
    ) -> "StandardSAE":
        state_dict = torch.load(f"{path}/ae.pt")
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)

        # Useful for analysis where e.g. feature activation magnitudes matter.
        # If trained with the April-update variant, decoder weights aren't
        # normalized, so this is skipped there.
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


@dataclass
class StandardSAEConfig(SAEConfig):
    l1_penalty: float = 1e-1
    sparsity_warmup_steps: Optional[int] = 2000
    resample_steps: Optional[int] = None


class StandardTrainer(SAETrainer):
    def __init__(self, steps: int, cfg: StandardSAEConfig):
        super().__init__(steps, cfg)
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps
        self.sparsity_warmup_steps = cfg.sparsity_warmup_steps
        self.l1_penalty = cfg.l1_penalty
        self.resample_steps = cfg.resample_steps

        self.ae = StandardSAE(cfg.activation_dim, cfg.dict_size)

        # Paper default; does NOT use the 1/sqrt(dict_size) auto-LR heuristic
        # even if cfg.lr is None, unlike the top-k trainers.
        self.lr = cfg.lr if cfg.lr is not None else 1e-3

        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = torch.zeros(self.ae.dict_size, dtype=torch.long)
        else:
            self.steps_since_active = None

        self.optimizer = ConstrainedAdam(
            self.ae.parameters(), self.ae.decoder.parameters(), lr=self.lr
        )

        lr_fn = get_lr_schedule(
            steps,
            cfg.warmup_steps,
            cfg.decay_start,
            self.resample_steps,
            cfg.sparsity_warmup_steps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            steps, cfg.sparsity_warmup_steps
        )

        self.logging_parameters = []

    def to(self, *args, **kwargs):
        self.ae.to(*args, **kwargs)
        if self.steps_since_active is not None:
            self.steps_since_active = self.steps_since_active.to(*args, **kwargs)

    def resample_neurons(self, deads: torch.Tensor, activations: torch.Tensor):
        with torch.no_grad():
            if deads.sum() == 0:
                return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = torch.multinomial(
                losses, num_samples=n_resample, replacement=False
            )
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:, deads] = (
                sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)
            ).T
            self.ae.encoder.bias[deads] = 0.0

            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()["state"]
            # encoder weight
            state_dict[1]["exp_avg"][deads] = 0.0
            state_dict[1]["exp_avg_sq"][deads] = 0.0
            # encoder bias
            state_dict[2]["exp_avg"][deads] = 0.0
            state_dict[2]["exp_avg_sq"][deads] = 0.0
            # decoder weight
            state_dict[3]["exp_avg"][:, deads] = 0.0
            state_dict[3]["exp_avg_sq"][:, deads] = 0.0

    def loss(self, x, step=None, logging=False):
        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "mse_loss": recon_loss.item(),
                    "sparsity_loss": l1_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x, _):
        self.optimizer.zero_grad()
        loss = self.loss(x, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, x)

        return loss.item()
