import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lapsum.topk import soft_topk
from torch.utils.data import DataLoader
from tqdm import tqdm

from classae.const import SUPPORTED_ARCHITECTURES, is_class_aligned
from classae.dataset import ActivationsDataset
from classae.eval.pmi import (
    compute_conditional_and_priors,
    compute_marginal_firing_rate,
    compute_pmi,
    normalize_pmi_to_unit_interval,
)

try:
    from classae.eval.posthoc_M import build_posthoc_M
    from classae.sae.core import topk_per_row

    _POSTHOC_M_AVAILABLE = True
except ImportError:
    _POSTHOC_M_AVAILABLE = False


# ======================================================================
# Data loading
# ======================================================================


def load_all_activations(activations_path: str, batch_size: int, num_workers: int):
    dataset = ActivationsDataset(activations_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    xs, ys = [], []
    for x, y in tqdm(loader, desc=f"Loading activations from {activations_path}"):
        xs.append(x.float())
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).long()


# ======================================================================
# Binary one-vs-rest partitions and probes
# ======================================================================


def build_binary_partition(
    labels: torch.Tensor,
    target_class: int,
    negative_ratio: float,
    seed: int,
    pool_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if pool_idx is not None:
        candidate_mask = torch.zeros(len(labels), dtype=torch.bool)
        candidate_mask[pool_idx] = True
    else:
        candidate_mask = torch.ones(len(labels), dtype=torch.bool)

    pos_idx = ((labels == target_class) & candidate_mask).nonzero(as_tuple=True)[0]
    neg_pool = ((labels != target_class) & candidate_mask).nonzero(as_tuple=True)[0]

    n_neg = min(int(round(len(pos_idx) * negative_ratio)), len(neg_pool))
    neg_idx = torch.from_numpy(rng.choice(neg_pool.numpy(), size=n_neg, replace=False))

    idx = torch.cat([pos_idx, neg_idx])
    y_bin = torch.cat(
        [
            torch.ones(len(pos_idx), dtype=torch.long),
            torch.zeros(n_neg, dtype=torch.long),
        ]
    )
    return idx, y_bin


def train_binary_probe(
    x: torch.Tensor,
    y_bin: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> nn.Linear:
    n, dim = x.shape
    probe = nn.Linear(dim, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    x = x.to(device)
    y_bin = y_bin.to(device).float()

    probe.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_batch, y_batch = x[idx], y_bin[idx]
            optimizer.zero_grad()
            logits = probe(x_batch).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


@torch.inference_mode()
def binary_probe_accuracy(
    probe: nn.Linear,
    x: torch.Tensor,
    y_bin: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> float:
    correct = torch.empty(len(x), dtype=torch.bool)
    for start in range(0, len(x), chunk_size):
        end = start + chunk_size
        logits = probe(x[start:end].to(device)).squeeze(-1)
        preds = (logits > 0).long().cpu()
        correct[start:end] = preds == y_bin[start:end]
    return correct.float().mean().item()


def compute_activation_reference_scale(
    model, x_ref, chunk_size, device, percentile=90.0, max_examples_for_estimate=50_000
) -> torch.Tensor:
    if len(x_ref) > max_examples_for_estimate:
        idx = torch.randperm(len(x_ref))[:max_examples_for_estimate]
        x_ref = x_ref[idx]
    d = model.dict_size
    z_chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(x_ref), chunk_size), desc="Estimating activation scale"
        ):
            z_chunks.append(
                model.encode(x_ref[start : start + chunk_size].to(device)).cpu()
            )
    z_all = torch.cat(z_chunks, dim=0)
    ref_scale = torch.ones(d)
    for i in range(d):
        nz = z_all[:, i]
        nz = nz[nz > 0]
        if len(nz) > 0:
            ref_scale[i] = torch.quantile(nz, percentile / 100.0)
    return ref_scale.clamp(min=1e-6)


def compute_native_selection_weights(model, x_batch: torch.Tensor) -> torch.Tensor:
    f, _, post_relu_acts = model.encode(x_batch, return_active=True)
    k_hat_sim = (f > 0).sum(dim=1).clamp_min(1).float()
    return soft_topk(post_relu_acts, k_hat_sim.unsqueeze(1), model.alpha.clone())


def build_strength_vector(z_batch, ref_scale, info_c, edit_strength_scale):
    act_norm = (z_batch / ref_scale.unsqueeze(0)).clamp(max=1.0)
    strength = act_norm * info_c.unsqueeze(0) * edit_strength_scale
    return strength.clamp(min=0.0, max=1.0)


def get_hard_ablation_matrix(model, precomputed_matrix, rho, device) -> torch.Tensor:
    if is_class_aligned(model) and precomputed_matrix is None:
        return model.calculate_M().detach().to(device)
    if precomputed_matrix is None:
        raise ValueError(
            "Model is not class-aligned, so --precomputed-matrix is required for "
            "--ablation-mode hard."
        )
    if not _POSTHOC_M_AVAILABLE:
        raise ImportError(
            "classae.eval.posthoc_M.build_posthoc_M / classae.sae.core.topk_per_row "
            "could not be imported -- check these still live at that path in your "
            "current package layout; this is the same post-hoc-M construction used "
            "by the original (non-graded) TPP script in this project."
        )
    train_A = torch.load(precomputed_matrix).to(device)
    M, k = build_posthoc_M(train_A, rho=rho)
    return topk_per_row(M, k).to(device)


def select_hard_ablation_features(
    M: torch.Tensor, concept: int, hard_topk: int | None
) -> torch.Tensor:
    col = M[:, concept]
    if hard_topk is None:
        return (col > 0).nonzero(as_tuple=True)[0]
    k = min(hard_topk, col.numel())
    return torch.topk(col, k).indices


@torch.inference_mode()
def compute_attribution_scores(
    model,
    probe: nn.Linear,
    x_partition: torch.Tensor,
    y_bin_partition: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    pos_mask = y_bin_partition == 1
    neg_mask = ~pos_mask

    z_sum_pos = torch.zeros(model.dict_size)
    z_sum_neg = torch.zeros(model.dict_size)
    n_pos, n_neg = 0, 0

    for start in range(0, len(x_partition), chunk_size):
        end = start + chunk_size
        z = model.encode(x_partition[start:end].to(device)).cpu()
        batch_pos = pos_mask[start:end]
        batch_neg = neg_mask[start:end]
        z_sum_pos += z[batch_pos].sum(dim=0)
        z_sum_neg += z[batch_neg].sum(dim=0)
        n_pos += int(batch_pos.sum().item())
        n_neg += int(batch_neg.sum().item())

    a_pos = z_sum_pos / max(n_pos, 1)
    a_neg = z_sum_neg / max(n_neg, 1)
    diff = a_pos - a_neg  # [d]

    try:
        decoder_weight = (
            model.decoder.weight.detach().cpu()
        )  # [activation_dim, dict_size]
    except AttributeError:
        decoder_weight = model.W_dec.T.detach().cpu()
    P = probe.weight.detach().cpu().squeeze(0)  # [activation_dim]
    projection = decoder_weight.t() @ P  # [dict_size], = d_a . P per latent a

    return projection * diff


def select_attribution_ablation_features(
    attribution_scores: torch.Tensor, topk: int
) -> torch.Tensor:
    """Top-N latents by SIGNED attribution score, per the paper -- not top-|N|."""
    k = min(topk, attribution_scores.numel())
    return torch.topk(attribution_scores, k).indices


@torch.inference_mode()
def compute_edited_activations(
    model,
    x_all: torch.Tensor,
    concept: int,
    mode: str,
    chunk_size: int,
    device: torch.device,
    # hard / attribution modes: the exact indices to zero, already
    # selected by the caller (select_hard_ablation_features or
    # select_attribution_ablation_features) -- both modes share this
    # same "zero these indices unconditionally" application, they only
    # differ in HOW the index set was chosen.
    feature_indices: torch.Tensor | None = None,
    # graded mode:
    ref_scale: torch.Tensor | None = None,
    info: torch.Tensor | None = None,
    edit_strength_scale: float = 1.0,
    use_native_activation_strength: bool = False,
) -> tuple[torch.Tensor, int | None]:
    out_chunks = []
    ablated_feature_count = None
    if mode in ("hard", "attribution"):
        ablated_feature_count = (
            len(feature_indices) if feature_indices is not None else 0
        )
        mask = torch.zeros(model.dict_size, device=device, dtype=torch.bool)
        if feature_indices is not None and len(feature_indices) > 0:
            mask[feature_indices.to(device)] = True

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        z = model.encode(x_batch)

        if mode in ("hard", "attribution"):
            z_edited = z.masked_fill(mask.unsqueeze(0), 0.0)
        else:
            info_c = info[:, concept].to(device)
            if use_native_activation_strength:
                activation_signal = compute_native_selection_weights(model, x_batch)
            else:
                activation_signal = z
            strength = build_strength_vector(
                activation_signal, ref_scale.to(device), info_c, edit_strength_scale
            )
            z_edited = z * (1.0 - strength)

        out_chunks.append(model.decode(z_edited).cpu())

    return torch.cat(out_chunks, dim=0), ablated_feature_count


# ======================================================================
# Main driver
# ======================================================================


def main(
    architecture: str,
    checkpoint_path: str,
    train_activations_path: str,
    test_activations_path: str,
    concepts: list[int] | None,
    num_random_concepts: int,
    negative_ratio: float,
    ablation_mode: str,
    precomputed_matrix: str | None,
    rho: float,
    hard_topk: int | None,
    attribution_topk: int,
    pmi_activations_path: str | None,
    no_empirical: bool,
    edit_strength_scale: float,
    pmi_upper_percentile: float,
    activation_scale_percentile: float,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    probe_batch_size: int,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    seed: int,
    output_path: str | None,
    device: str | None,
    max_train_examples: int | None,
    max_test_examples: int | None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path, device=device
    )
    model.eval()

    if no_empirical and ablation_mode != "graded":
        raise ValueError("--no-empirical only applies to --ablation-mode graded.")
    if no_empirical and not is_class_aligned(model):
        raise ValueError("--no-empirical requires a class-aligned SAE architecture.")

    print(
        "Loading probe-training activations (clean, never touched by the SAE again)..."
    )
    x_train, y_train = load_all_activations(
        train_activations_path, batch_size, num_workers
    )
    if max_train_examples is not None:
        x_train, y_train = x_train[:max_train_examples], y_train[:max_train_examples]

    print("Loading evaluation activations...")
    x_test, y_test = load_all_activations(
        test_activations_path, batch_size, num_workers
    )
    if max_test_examples is not None:
        x_test, y_test = x_test[:max_test_examples], y_test[:max_test_examples]

    num_classes = int(y_train.max().item()) + 1

    # ------------------------------------------------------------
    # Concept pool (m classes)
    # ------------------------------------------------------------
    m_concepts = list(concepts) if concepts is not None else []
    if num_random_concepts > 0:
        pool = [c for c in range(num_classes) if c not in m_concepts]
        m_concepts = m_concepts + rng.sample(pool, min(num_random_concepts, len(pool)))
    print(f"Concepts (m={len(m_concepts)}): {m_concepts}")

    # ------------------------------------------------------------
    # Ablation-set construction: hard M or empirical/native informativeness
    # ------------------------------------------------------------
    M = None
    ref_scale = None
    info = None
    if ablation_mode == "hard":
        M = get_hard_ablation_matrix(model, precomputed_matrix, rho, device)
    elif ablation_mode == "attribution":
        pass
    else:  # graded
        if no_empirical:
            print(
                "Using the model's own affinity matrix M as informativeness and its "
                "native selection weights as activation strength (--no-empirical)."
            )
            info = model.calculate_M().detach().cpu()
            ref_scale = torch.ones(model.dict_size)
        else:
            if pmi_activations_path is not None:
                x_pmi, y_pmi = load_all_activations(
                    pmi_activations_path, batch_size, num_workers
                )
            else:
                print(
                    "[note] --pmi-activations-path not given; reusing the probe-training "
                    "split for PMI / activation-scale estimation."
                )
                x_pmi, y_pmi = x_train, y_train
            p_fire_given_c, class_priors = compute_conditional_and_priors(
                model, x_pmi, y_pmi, num_classes, chunk_size, device
            )
            marginal_rate = compute_marginal_firing_rate(p_fire_given_c, class_priors)
            pmi = compute_pmi(p_fire_given_c, marginal_rate)
            info = normalize_pmi_to_unit_interval(
                pmi, upper_percentile=pmi_upper_percentile
            )
            ref_scale = compute_activation_reference_scale(
                model, x_pmi, chunk_size, device, percentile=activation_scale_percentile
            )

    # ------------------------------------------------------------
    # Per-concept binary partitions + independent probes
    # ------------------------------------------------------------
    probes = {}
    test_partitions = {}  # concept -> (idx into x_test, y_bin)
    baseline_acc = {}  # concept -> A_j (SAE round-trip, clean)
    attribution_scores = (
        {}
    )  # concept -> [dict_size] signed score, attribution mode only

    selection_pool, eval_pool = None, None
    if ablation_mode == "attribution":
        perm = np.random.default_rng(seed).permutation(len(x_train))
        half = len(perm) // 2
        selection_pool = torch.from_numpy(perm[:half])
        eval_pool = torch.from_numpy(perm[half:])
        print(
            f"[note] --ablation-mode attribution: splitting train activations into "
            f"disjoint selection pool (n={len(selection_pool)}) and evaluation pool "
            f"(n={len(eval_pool)}) -- the probe used to pick which latents to ablate "
            f"is never the same probe, or trained on the same data, as the probe "
            f"used to measure the resulting accuracy drop."
        )

    for j in tqdm(m_concepts, desc="Training binary probes"):
        train_idx, train_y_bin = build_binary_partition(
            y_train, j, negative_ratio, seed=seed * 100_003 + j, pool_idx=eval_pool
        )
        probe_j = train_binary_probe(
            x_train[train_idx],
            train_y_bin,
            epochs=probe_epochs,
            lr=probe_lr,
            weight_decay=probe_weight_decay,
            batch_size=probe_batch_size,
            device=device,
        )
        probes[j] = probe_j

        if ablation_mode == "attribution":
            sel_idx, sel_y_bin = build_binary_partition(
                y_train,
                j,
                negative_ratio,
                seed=seed * 100_037 + j,
                pool_idx=selection_pool,
            )
            probe_select_j = train_binary_probe(
                x_train[sel_idx],
                sel_y_bin,
                epochs=probe_epochs,
                lr=probe_lr,
                weight_decay=probe_weight_decay,
                batch_size=probe_batch_size,
                device=device,
            )
            attribution_scores[j] = compute_attribution_scores(
                model, probe_select_j, x_train[sel_idx], sel_y_bin, chunk_size, device
            )

        test_idx, test_y_bin = build_binary_partition(
            y_test, j, negative_ratio, seed=seed * 100_019 + j
        )
        test_partitions[j] = (test_idx, test_y_bin)

        with torch.inference_mode():
            x_roundtrip = model.decode(model.encode(x_test[test_idx].to(device))).cpu()
        baseline_acc[j] = binary_probe_accuracy(
            probe_j, x_roundtrip, test_y_bin, device, chunk_size
        )

    A = {i: {} for i in m_concepts}
    ablated_counts = {}
    for i in tqdm(m_concepts, desc="Ablating each concept, evaluating all probes"):
        feature_indices = None
        if ablation_mode == "hard":
            feature_indices = select_hard_ablation_features(M, i, hard_topk)
        elif ablation_mode == "attribution":
            feature_indices = select_attribution_ablation_features(
                attribution_scores[i], attribution_topk
            )

        x_hat_i, n_ablated = compute_edited_activations(
            model,
            x_test,
            i,
            ablation_mode,
            chunk_size,
            device,
            feature_indices=feature_indices,
            ref_scale=ref_scale,
            info=info,
            edit_strength_scale=edit_strength_scale,
            use_native_activation_strength=no_empirical,
        )
        ablated_counts[i] = n_ablated
        for j in m_concepts:
            test_idx, test_y_bin = test_partitions[j]
            A[i][j] = binary_probe_accuracy(
                probes[j], x_hat_i[test_idx], test_y_bin, device, chunk_size
            )

    if ablation_mode in ("hard", "attribution"):
        counts = list(ablated_counts.values())
        print(
            f"\n[diagnostic] ablation feature counts per concept ({ablation_mode}): "
            f"mean={np.mean(counts):.1f}, min={min(counts)}, max={max(counts)} "
            f"(out of dict_size={model.dict_size}) -- if this is a large fraction "
            f"of dict_size, on-diagonal and off-diagonal deltas will look identical "
            f"regardless of concept, since the ablation is large enough to degrade "
            f"reconstruction uniformly rather than concept-specifically."
        )

    # ------------------------------------------------------------
    # STPP score
    # ------------------------------------------------------------
    diag_deltas = [A[i][i] - baseline_acc[i] for i in m_concepts]
    off_diag_deltas = [
        A[i][j] - baseline_acc[j] for i in m_concepts for j in m_concepts if i != j
    ]

    mean_diag_delta = float(np.mean(diag_deltas)) if diag_deltas else None
    mean_off_diag_delta = float(np.mean(off_diag_deltas)) if off_diag_deltas else None
    stpp = (
        mean_diag_delta - mean_off_diag_delta
        if mean_diag_delta is not None and mean_off_diag_delta is not None
        else None
    )

    print("\n=== STPP (original-paper formula) ===")
    print(f"m concepts:                                  {len(m_concepts)}")
    print(
        f"Mean baseline accuracy A_j:                   {np.mean(list(baseline_acc.values())):.4f}"
    )
    print(f"Mean on-diagonal delta  (A_ii - A_i):          {mean_diag_delta:.4f}")
    print(f"Mean off-diagonal delta (A_ij - A_j, i!=j):    {mean_off_diag_delta:.4f}")
    print(f"STPP = mean(on-diag delta) - mean(off-diag delta):  {stpp:.4f}")

    summary = {
        "architecture": architecture,
        "ablation_mode": ablation_mode,
        "no_empirical": no_empirical,
        "num_concepts": len(m_concepts),
        "concepts": m_concepts,
        "negative_ratio": negative_ratio,
        "baseline_accuracy_per_concept": baseline_acc,
        "ablation_grid_A": A,
        "mean_on_diagonal_delta": mean_diag_delta,
        "mean_off_diagonal_delta": mean_off_diag_delta,
        "stpp": stpp,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved results to {output_path}")

    return summary


def cli():
    parser = argparse.ArgumentParser(
        description="STPP eval (original-paper formula): independent binary probes, "
        "m x m ablation cross-grid."
    )
    parser.add_argument(
        "--architecture",
        "-a",
        required=True,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-activations-path", required=True)
    parser.add_argument("--test-activations-path", required=True)

    parser.add_argument("--concepts", type=int, nargs="+", default=None)
    parser.add_argument("--num-random-concepts", type=int, default=20)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Negatives per positive when building each concept's binary partition. "
        "1.0 = balanced binary classification.",
    )

    parser.add_argument(
        "--ablation-mode", choices=["hard", "graded", "attribution"], default="graded"
    )
    parser.add_argument(
        "--precomputed-matrix",
        default=None,
        help="Only used with --ablation-mode hard on a non-class-aligned architecture.",
    )
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument(
        "--hard-topk",
        type=int,
        default=None,
        help="Hard mode only: ablate the top-K entries of M[:,concept] by magnitude, "
        "independent of rho. If unset, falls back to the literal 'M[:,concept] > 0' "
        "rule -- at large dict_size / rho this can ablate hundreds or thousands of "
        "features and swamp any concept-specific signal with uniform collateral "
        "damage (see the printed diagnostic after a hard-mode run). Recommended: "
        "start around 20-50 and check the STPP score is no longer ~0.",
    )
    parser.add_argument(
        "--attribution-topk",
        type=int,
        default=50,
        help="Attribution mode only: number of latents to ablate per concept, "
        "selected by the original TPP-paper attribution score I(a,c) = (d_a . P) * "
        "(a_pos - a_neg) -- see module docstring for the derivation. Ablates the "
        "top-N by SIGNED score, not top-|N|.",
    )

    parser.add_argument("--pmi-activations-path", default=None)
    parser.add_argument(
        "--no-empirical",
        action="store_true",
        help="Graded mode only, class-aligned architectures only: use the model's own "
        "M and native selection weights instead of an empirical PMI pass.",
    )
    parser.add_argument("--edit-strength-scale", type=float, default=1.0)
    parser.add_argument("--pmi-upper-percentile", type=float, default=99.0)
    parser.add_argument("--activation-scale-percentile", type=float, default=90.0)

    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=256)

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)

    args = parser.parse_args()
    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        train_activations_path=args.train_activations_path,
        test_activations_path=args.test_activations_path,
        concepts=args.concepts,
        num_random_concepts=args.num_random_concepts,
        negative_ratio=args.negative_ratio,
        ablation_mode=args.ablation_mode,
        precomputed_matrix=args.precomputed_matrix,
        rho=args.rho,
        hard_topk=args.hard_topk,
        attribution_topk=args.attribution_topk,
        pmi_activations_path=args.pmi_activations_path,
        no_empirical=args.no_empirical,
        edit_strength_scale=args.edit_strength_scale,
        pmi_upper_percentile=args.pmi_upper_percentile,
        activation_scale_percentile=args.activation_scale_percentile,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_weight_decay=args.probe_weight_decay,
        probe_batch_size=args.probe_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        output_path=args.output_path,
        device=args.device,
        max_train_examples=args.max_train_examples,
        max_test_examples=args.max_test_examples,
    )


if __name__ == "__main__":
    cli()
