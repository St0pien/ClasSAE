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
from classae.sae.classae import ClasSAE

try:
    from scipy.stats import pearsonr, wilcoxon

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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
# Empirical PMI(i, c) -- shared with the standalone PMI eval script;
# see ca_sae.eval.pmi.
# ======================================================================


def compute_activation_reference_scale(
    model,
    x_ref: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    percentile: float = 90.0,
    max_examples_for_estimate: int = 50_000,
) -> torch.Tensor:
    if len(x_ref) > max_examples_for_estimate:
        idx = torch.randperm(len(x_ref))[:max_examples_for_estimate]
        x_ref = x_ref[idx]

    d = model.dict_size
    z_chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(x_ref), chunk_size),
            desc="Estimating per-feature activation scale",
        ):
            z_chunks.append(
                model.encode(x_ref[start : start + chunk_size].to(device)).cpu()
            )
    z_all = torch.cat(z_chunks, dim=0)  # [min(N, max_examples_for_estimate), d]

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


# ======================================================================
# Building and applying the graded edit
# ======================================================================


def build_strength_vector(
    z_batch: torch.Tensor,
    ref_scale: torch.Tensor,
    info_c: torch.Tensor,
    edit_strength_scale: float,
) -> torch.Tensor:
    act_norm = (z_batch / ref_scale.unsqueeze(0)).clamp(max=1.0)
    strength = act_norm * info_c.unsqueeze(0) * edit_strength_scale
    return strength.clamp(min=0.0, max=1.0)


@torch.inference_mode()
def graded_edit_and_probe(
    model,
    probe: nn.Linear,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    ref_scale: torch.Tensor,
    info_c: torch.Tensor,
    target_class: int,
    edit_strength_scale: float,
    chunk_size: int,
    device: torch.device,
    use_native_activation_strength: bool = False,
) -> dict:
    ref_scale = ref_scale.to(device)
    info_c = info_c.to(device)

    target_mask = labels_all == target_class
    retain_mask = ~target_mask

    correct = torch.empty(len(x_all), dtype=torch.bool)
    target_logit_after = torch.empty(len(x_all))
    mean_strength_applied = torch.empty(len(x_all))
    effective_features_edited = torch.empty(len(x_all))

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        z = model.encode(x_batch)
        if use_native_activation_strength:
            activation_signal = compute_native_selection_weights(model, x_batch)
        else:
            activation_signal = z
        strength = build_strength_vector(
            activation_signal, ref_scale, info_c, edit_strength_scale
        )
        z_edited = z * (1.0 - strength)
        x_hat = model.decode(z_edited)

        logits = probe(x_hat)
        preds = logits.argmax(dim=1)

        correct[start:end] = (preds == labels_batch).cpu()
        target_logit_after[start:end] = logits[:, target_class].cpu()
        mean_strength_applied[start:end] = strength.mean(dim=1).cpu()
        effective_features_edited[start:end] = strength.sum(dim=1).cpu()

    return {
        "forget_accuracy": correct[target_mask].float().mean().item(),
        "retain_accuracy": correct[retain_mask].float().mean().item(),
        "mean_target_logit_after": target_logit_after[target_mask].mean().item(),
        "mean_strength_applied": mean_strength_applied.mean().item(),
        "mean_effective_features_edited": effective_features_edited.mean().item(),
    }


@torch.inference_mode()
def probe_baseline(
    probe: nn.Linear,
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    target_class: int,
    chunk_size: int,
    device: torch.device,
    round_trip_through_sae: bool,
) -> dict:
    target_mask = labels_all == target_class
    retain_mask = ~target_mask
    correct = torch.empty(len(x_all), dtype=torch.bool)
    target_logit = torch.empty(len(x_all))

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        if round_trip_through_sae:
            z = model.encode(x_batch)
            x_used = model.decode(z)
        else:
            x_used = x_batch

        logits = probe(x_used)
        preds = logits.argmax(dim=1)
        correct[start:end] = (preds == labels_batch).cpu()
        target_logit[start:end] = logits[:, target_class].cpu()

    return {
        "forget_accuracy": correct[target_mask].float().mean().item(),
        "retain_accuracy": correct[retain_mask].float().mean().item(),
        "mean_target_logit_after": target_logit[target_mask].mean().item(),
    }


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> nn.Linear:
    n, dim = x_train.shape
    probe = nn.Linear(dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    probe.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, total_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_batch, y_batch = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = probe(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        print(
            f"  probe epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / n:.4f} acc={total_correct / n:.4f}"
        )

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


# ======================================================================
# Aggregate summary
# ======================================================================


def compute_aggregate_metrics(results: dict) -> dict:
    classes = sorted(results.keys())
    baseline_acc = np.array(
        [results[c]["baseline_sae_roundtrip"]["forget_accuracy"] for c in classes]
    )
    baseline_retain = np.array(
        [results[c]["baseline_sae_roundtrip"]["retain_accuracy"] for c in classes]
    )
    graded_forget = np.array(
        [results[c]["graded_edit"]["forget_accuracy"] for c in classes]
    )
    graded_retain = np.array(
        [results[c]["graded_edit"]["retain_accuracy"] for c in classes]
    )
    shuffled_forget = np.array(
        [results[c]["shuffled_info_control"]["forget_accuracy"] for c in classes]
    )
    shuffled_retain = np.array(
        [results[c]["shuffled_info_control"]["retain_accuracy"] for c in classes]
    )
    mean_strength = np.array(
        [results[c]["graded_edit"]["mean_strength_applied"] for c in classes]
    )
    mean_eff_features_graded = np.array(
        [results[c]["graded_edit"]["mean_effective_features_edited"] for c in classes]
    )
    mean_eff_features_shuffled = np.array(
        [
            results[c]["shuffled_info_control"]["mean_effective_features_edited"]
            for c in classes
        ]
    )

    graded_drop = baseline_acc - graded_forget
    shuffled_drop = baseline_acc - shuffled_forget
    paired_diff = graded_drop - shuffled_drop

    win_rate = float(np.mean(paired_diff > 0)) if len(paired_diff) > 0 else None

    wilcoxon_stat, wilcoxon_p = None, None
    corr_r, corr_p = None, None
    if _SCIPY_AVAILABLE:
        if len(paired_diff) >= 2 and not np.allclose(graded_drop, shuffled_drop):
            wilcoxon_stat, wilcoxon_p = wilcoxon(graded_drop, shuffled_drop)
            wilcoxon_stat, wilcoxon_p = float(wilcoxon_stat), float(wilcoxon_p)
        if (
            len(baseline_acc) >= 2
            and np.std(baseline_acc) > 0
            and np.std(graded_drop) > 0
        ):
            corr_r, corr_p = pearsonr(baseline_acc, graded_drop)
            corr_r, corr_p = float(corr_r), float(corr_p)
    elif (
        len(baseline_acc) >= 2 and np.std(baseline_acc) > 0 and np.std(graded_drop) > 0
    ):
        corr_r = float(np.corrcoef(baseline_acc, graded_drop)[0, 1])

    return {
        "num_classes": len(classes),
        "mean_edit_strength_applied": (
            float(mean_strength.mean()) if len(mean_strength) else None
        ),
        "mean_effective_features_edited_graded": (
            float(mean_eff_features_graded.mean())
            if len(mean_eff_features_graded)
            else None
        ),
        "mean_effective_features_edited_shuffled": (
            float(mean_eff_features_shuffled.mean())
            if len(mean_eff_features_shuffled)
            else None
        ),
        "forget_drop_per_effective_feature_graded": (
            float(np.mean(graded_drop / np.clip(mean_eff_features_graded, 1e-6, None)))
            if len(mean_eff_features_graded)
            else None
        ),
        "mean_forget_accuracy_drop_graded_edit": (
            float(graded_drop.mean()) if len(graded_drop) else None
        ),
        "std_forget_accuracy_drop_graded_edit": (
            float(graded_drop.std()) if len(graded_drop) else None
        ),
        "mean_forget_accuracy_drop_shuffled_control": (
            float(shuffled_drop.mean()) if len(shuffled_drop) else None
        ),
        "std_forget_accuracy_drop_shuffled_control": (
            float(shuffled_drop.std()) if len(shuffled_drop) else None
        ),
        "mean_paired_difference_graded_minus_shuffled": (
            float(paired_diff.mean()) if len(paired_diff) else None
        ),
        "fraction_of_classes_graded_beats_shuffled": win_rate,
        "wilcoxon_statistic_graded_vs_shuffled": wilcoxon_stat,
        "wilcoxon_p_value_graded_vs_shuffled": wilcoxon_p,
        "mean_retain_accuracy_baseline": (
            float(baseline_retain.mean()) if len(baseline_retain) else None
        ),
        "mean_retain_accuracy_graded_edit": (
            float(graded_retain.mean()) if len(graded_retain) else None
        ),
        "mean_retain_accuracy_shuffled_control": (
            float(shuffled_retain.mean()) if len(shuffled_retain) else None
        ),
        "mean_retain_accuracy_collateral_graded": (
            float((graded_retain - baseline_retain).mean())
            if len(graded_retain)
            else None
        ),
        "mean_retain_accuracy_collateral_shuffled": (
            float((shuffled_retain - baseline_retain).mean())
            if len(shuffled_retain)
            else None
        ),
        "pearson_corr_baseline_accuracy_vs_graded_drop": corr_r,
        "pearson_corr_p_value": corr_p,
        "scipy_available": _SCIPY_AVAILABLE,
    }


def print_aggregate_summary(agg: dict) -> None:
    print("\n=== Aggregate summary (graded PMI-weighted edit) ===")
    print(f"Classes evaluated:                                {agg['num_classes']}")
    print(
        f"Mean edit strength applied (both conditions):     {agg['mean_edit_strength_applied']:.4f}"
    )
    print(
        f"Mean effective features edited, graded / shuffled: "
        f"{agg['mean_effective_features_edited_graded']:.2f} / "
        f"{agg['mean_effective_features_edited_shuffled']:.2f}"
    )
    print(
        f"Forget-drop per effective feature edited (graded): {agg['forget_drop_per_effective_feature_graded']:.4f}"
    )
    print(
        f"Mean forget-acc drop, graded edit:                "
        f"{agg['mean_forget_accuracy_drop_graded_edit']:.3f} "
        f"(std {agg['std_forget_accuracy_drop_graded_edit']:.3f})"
    )
    print(
        f"Mean forget-acc drop, shuffled-info control:      "
        f"{agg['mean_forget_accuracy_drop_shuffled_control']:.3f} "
        f"(std {agg['std_forget_accuracy_drop_shuffled_control']:.3f})"
    )
    print(
        f"Mean paired difference (graded - shuffled):       {agg['mean_paired_difference_graded_minus_shuffled']:.3f}"
    )
    print(
        f"Fraction of classes where graded beats shuffled:  {agg['fraction_of_classes_graded_beats_shuffled']:.2f}"
    )
    if agg["wilcoxon_p_value_graded_vs_shuffled"] is not None:
        print(
            f"Wilcoxon signed-rank (graded vs shuffled):         "
            f"stat={agg['wilcoxon_statistic_graded_vs_shuffled']:.3f} "
            f"p={agg['wilcoxon_p_value_graded_vs_shuffled']:.4f}"
        )
    else:
        print(
            "Wilcoxon signed-rank (graded vs shuffled):         (install scipy, or drops were identical)"
        )
    print(
        f"Mean retain-accuracy collateral, graded / shuffled: "
        f"{agg['mean_retain_accuracy_collateral_graded']:+.4f} / "
        f"{agg['mean_retain_accuracy_collateral_shuffled']:+.4f}"
    )
    if agg["pearson_corr_baseline_accuracy_vs_graded_drop"] is not None:
        p_str = (
            f"p={agg['pearson_corr_p_value']:.4f}"
            if agg["pearson_corr_p_value"] is not None
            else "(p n/a)"
        )
        print(
            f"Corr(baseline accuracy, graded drop):             r={agg['pearson_corr_baseline_accuracy_vs_graded_drop']:.3f} {p_str}"
        )


# ======================================================================
# Main driver
# ======================================================================


def main(
    architecture: str,
    checkpoint_path: str,
    train_activations_path: str,
    test_activations_path: str,
    pmi_activations_path: str | None,
    no_empirical: bool,
    target_classes: list[int] | None,
    num_random_target_classes: int,
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

    if no_empirical and not is_class_aligned(model):
        raise ValueError(
            "--no-empirical requires a class-aligned SAE architecture "
            f"(got --architecture={architecture!r}, is_class_aligned(model) is False)."
        )

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
    # Informativeness info[i, c] and per-feature activation reference
    # scale ref_scale[i]. Either estimated empirically on held-out
    # reference data, or, for class-aligned SAEs, read directly off the
    # model (--no-empirical): the feature-class affinity
    # matrix M from calculate_M() stands in for PMI, and an all-ones
    # ref_scale is used because the native selection weights plugged in
    # as the activation signal (see compute_native_selection_weights)
    # are already bounded in (0,1) by construction.
    # ------------------------------------------------------------
    if no_empirical:
        if pmi_activations_path is not None:
            print(
                "[note] --no-empirical is set; ignoring "
                "--pmi-activations-path -- no empirical estimation pass is run."
            )
        print(
            "Using the model's own feature-class affinity matrix M (calculate_M()) "
            "as informativeness, and its native soft Top-K selection weights as the "
            "per-sample activation-strength signal -- no empirical estimation pass."
        )
        info = model.calculate_M().detach().cpu()  # [d, C]
        if info.shape[1] != num_classes:
            raise ValueError(
                f"model.calculate_M() has {info.shape[1]} classes but the loaded "
                f"labels imply {num_classes}; refusing to silently misalign columns."
            )
        ref_scale = torch.ones(model.dict_size)
    else:
        if pmi_activations_path is not None:
            print(
                "Loading separate activations for PMI / activation-scale estimation..."
            )
            x_pmi, y_pmi = load_all_activations(
                pmi_activations_path, batch_size, num_workers
            )
        else:
            print(
                "[note] --pmi-activations-path not given; reusing the probe-training "
                "split for PMI and activation-scale estimation. This does not leak "
                "test labels, but if you want PMI estimated fully independently of "
                "anything the probe saw, pass a third, disjoint split explicitly."
            )
            x_pmi, y_pmi = x_train, y_train

        p_fire_given_c, class_priors = compute_conditional_and_priors(
            model, x_pmi, y_pmi, num_classes, chunk_size, device
        )
        marginal_rate = compute_marginal_firing_rate(p_fire_given_c, class_priors)
        pmi = compute_pmi(p_fire_given_c, marginal_rate)
        info = normalize_pmi_to_unit_interval(
            pmi, upper_percentile=pmi_upper_percentile
        )  # [d, C]

        ref_scale = compute_activation_reference_scale(
            model, x_pmi, chunk_size, device, percentile=activation_scale_percentile
        )  # [d]

    # ------------------------------------------------------------
    # Train and freeze the independent probe
    # ------------------------------------------------------------
    print("Training independent linear probe on clean embeddings...")
    probe = train_linear_probe(
        x_train,
        y_train,
        num_classes,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        batch_size=probe_batch_size,
        device=device,
    )

    # ------------------------------------------------------------
    # Target classes
    # ------------------------------------------------------------
    targets = list(target_classes) if target_classes is not None else []
    if num_random_target_classes > 0:
        pool = [c for c in range(num_classes) if c not in targets]
        targets = targets + rng.sample(pool, min(num_random_target_classes, len(pool)))
    print(f"Target classes: {targets}")

    # ------------------------------------------------------------
    # Per-class graded-edit sweep
    # ------------------------------------------------------------
    results = {}
    d = model.dict_size
    for c in tqdm(targets, desc="Graded probe perturbation"):
        info_c = info[:, c]
        mask = info_c > 0
        info_c_shuffled = torch.zeros_like(info_c)
        vals = info_c[mask]
        info_c_shuffled[mask] = vals[
            torch.randperm(
                vals.numel(), generator=torch.Generator().manual_seed(seed + c)
            )
        ]

        baseline_raw = probe_baseline(
            probe,
            model,
            x_test,
            y_test,
            c,
            chunk_size,
            device,
            round_trip_through_sae=False,
        )
        baseline_roundtrip = probe_baseline(
            probe,
            model,
            x_test,
            y_test,
            c,
            chunk_size,
            device,
            round_trip_through_sae=True,
        )

        graded_result = graded_edit_and_probe(
            model,
            probe,
            x_test,
            y_test,
            ref_scale,
            info_c,
            c,
            edit_strength_scale,
            chunk_size,
            device,
            use_native_activation_strength=no_empirical,
        )
        shuffled_result = graded_edit_and_probe(
            model,
            probe,
            x_test,
            y_test,
            ref_scale,
            info_c_shuffled,
            c,
            edit_strength_scale,
            chunk_size,
            device,
            use_native_activation_strength=no_empirical,
        )

        results[c] = {
            "baseline_raw_embeddings": baseline_raw,
            "baseline_sae_roundtrip": baseline_roundtrip,
            "graded_edit": graded_result,
            "shuffled_info_control": shuffled_result,
        }

        print(
            f"  class {c}: baseline(raw/roundtrip) forget_acc="
            f"{baseline_raw['forget_accuracy']:.3f}/{baseline_roundtrip['forget_accuracy']:.3f} | "
            f"graded_edit forget_acc={graded_result['forget_accuracy']:.3f} "
            f"retain_acc={graded_result['retain_accuracy']:.3f} "
            f"eff_features_edited={graded_result['mean_effective_features_edited']:.2f} | "
            f"shuffled_control forget_acc={shuffled_result['forget_accuracy']:.3f} "
            f"retain_acc={shuffled_result['retain_accuracy']:.3f} "
            f"eff_features_edited={shuffled_result['mean_effective_features_edited']:.2f}"
        )

    aggregate = compute_aggregate_metrics(results)
    print_aggregate_summary(aggregate)
    if not _SCIPY_AVAILABLE:
        print(
            "\n[note] scipy not found -- Wilcoxon test skipped. `pip install scipy` for the full stats."
        )

    summary = {
        "architecture": architecture,
        "num_target_classes": len(results),
        "edit_strength_scale": edit_strength_scale,
        "no_empirical": no_empirical,
        "pmi_upper_percentile": pmi_upper_percentile,
        "activation_scale_percentile": activation_scale_percentile,
        "aggregate": aggregate,
        "results_by_class": results,
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
        description="Graded, PMI-weighted probe perturbation evaluation for SoftSAE-CA."
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
    parser.add_argument(
        "--pmi-activations-path",
        default=None,
        help="Optional separate split for estimating PMI(i,c) and per-feature activation "
        "reference scale. Defaults to reusing --train-activations-path if not given. "
        "Ignored when --no-empirical is set.",
    )
    parser.add_argument(
        "--no-empirical",
        action="store_true",
        help="For class-aligned SAEs only (is_class_aligned(model)): skip the empirical "
        "PMI and activation-reference-scale estimation pass entirely, and instead build "
        "the graded edit purely from information the model already carries -- the "
        "feature-class affinity matrix M (model.calculate_M(), standing in for PMI) and "
        "the model's own native soft Top-K selection weights per sample (standing in for "
        "the empirically-normalized activation strength). Raises if the loaded "
        "architecture is not class-aligned.",
    )
    parser.add_argument("--target-classes", type=int, nargs="+", default=None)
    parser.add_argument("--num-random-target-classes", type=int, default=20)
    parser.add_argument(
        "--edit-strength-scale",
        type=float,
        default=1.0,
        help="Global multiplier on the strength vector before clamping to [0,1]. "
        "Values > 1 push more (activation, informativeness) pairs toward full ablation.",
    )
    parser.add_argument("--pmi-upper-percentile", type=float, default=99.0)
    parser.add_argument("--activation-scale-percentile", type=float, default=90.0)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
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
        pmi_activations_path=args.pmi_activations_path,
        no_empirical=args.no_empirical,
        target_classes=args.target_classes,
        num_random_target_classes=args.num_random_target_classes,
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
