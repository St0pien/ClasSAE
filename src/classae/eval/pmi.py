import torch
from tqdm import tqdm


@torch.inference_mode()
def compute_conditional_and_priors(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    num_classes: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = model.dict_size
    fire_count = torch.zeros(d, num_classes)
    class_count = torch.zeros(num_classes)

    for start in tqdm(range(0, len(x_all), chunk_size), desc="Streaming PMI stats"):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        z = model.encode(x_batch)
        fired = (z > 0).float()  # [B, d], one chunk at a time -- never the full N

        onehot = torch.zeros(len(labels_batch), num_classes, device=device)
        onehot.scatter_(1, labels_batch.unsqueeze(1), 1.0)

        fire_count += (fired.T @ onehot).cpu()
        class_count += onehot.sum(dim=0).cpu()

    p_fire_given_c = fire_count / class_count.clamp(min=1).unsqueeze(0)
    class_priors = class_count / class_count.sum().clamp(min=1)
    return p_fire_given_c, class_priors


def compute_marginal_firing_rate(
    p_fire_given_c: torch.Tensor, class_priors: torch.Tensor
) -> torch.Tensor:
    return p_fire_given_c @ class_priors


def compute_pmi(
    p_fire_given_c: torch.Tensor,
    marginal_rate: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p_cond = p_fire_given_c.clamp(min=eps)
    p_marg = marginal_rate.clamp(min=eps).unsqueeze(1)
    return torch.log(p_cond / p_marg)


def normalized_pmi(
    pmi: torch.Tensor,
    p_fire_given_c: torch.Tensor,
    class_priors: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p_joint = (p_fire_given_c * class_priors.unsqueeze(0)).clamp(min=eps)
    return pmi / (-torch.log(p_joint))


def normalize_pmi_to_unit_interval(
    pmi: torch.Tensor, upper_percentile: float = 99.0
) -> torch.Tensor:
    pos = pmi.clamp(min=0)
    ref = torch.quantile(pos.flatten(), upper_percentile / 100.0).clamp(min=1e-6)
    return (pos / ref).clamp(max=1.0)
