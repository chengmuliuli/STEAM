#!/usr/bin/env python3
"""Evaluate an 8/32-bin STEAM checkpoint on a fixed deterministic pair set.

This is a post-training quality gate, not just a training-log parser.  It uses
the production PairDataset/collator, fixes temporal sampling to epoch 0, loads
the saved critic checkpoint, and reports:

* exact-bin and +/-1-neighbor accuracy;
* directional (progress/regress) accuracy;
* NLL cross entropy and predictive entropy;
* target and prediction bin histograms;
* uniform and empirical-prior/constant-predictor baselines.

The command exits non-zero unless the model improves over data-only constant
baselines by configurable margins.  This prevents a formally completed 32-bin
run from silently flowing into advantage labeling while still behaving like a
random/prior-only classifier.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Ensure deterministic PairDataset monkeypatches are active even when the
# caller forgot to rely on Python's sitecustomize auto-import.
import steam_runtime_compat  # noqa: F401,E402

from rlinf.data.datasets.steam import BinaryPairDataCollator, PairDataset  # noqa: E402
from rlinf.models.embodiment.value_model.steam.modeling_critic import (  # noqa: E402
    SteamCriticModel,
)


def move_to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def build_dataset(entry: dict, cfg, seed: int) -> PairDataset:
    return PairDataset(
        dataset_path=str(entry["dataset_path"]),
        camera_keys=tuple(cfg.data.camera_keys),
        k=int(cfg.data.k),
        dataset_type=str(entry.get("type", cfg.data.get("dataset_type", "rollout"))),
        only_success=bool(entry.get("only_success", cfg.data.get("only_success", False))),
        min_episode_length=cfg.data.get("min_episode_length", None),
        num_bins=int(cfg.actor.model.num_bins),
        length_scale_enabled=bool(
            entry.get(
                "length_scale_enabled",
                cfg.data.get("length_scale_enabled", False),
            )
        ),
        length_scale_percentile=float(
            entry.get(
                "length_scale_percentile",
                cfg.data.get("length_scale_percentile", 90.0),
            )
        ),
        seed=seed,
    )


def fixed_indices(dataset: PairDataset, max_samples: int) -> list[int]:
    """Evenly cover temporal anchors while preserving +/- pairs."""
    pair_count = int(dataset.num_pair_positions)
    if pair_count < 1:
        return []
    wanted_pairs = max(1, min(pair_count, max_samples // 2))
    if wanted_pairs == pair_count:
        positions = np.arange(pair_count, dtype=np.int64)
    else:
        positions = np.unique(
            np.linspace(0, pair_count - 1, num=wanted_pairs, dtype=np.int64)
        )
    indices: list[int] = []
    for pos in positions.tolist():
        indices.extend((2 * int(pos), 2 * int(pos) + 1))
    return indices[:max_samples]


def baseline_metrics(targets: np.ndarray, num_bins: int) -> dict[str, float]:
    counts = np.bincount(targets, minlength=num_bins).astype(np.float64)
    probs = counts / max(1.0, counts.sum())
    nonzero = probs > 0
    empirical_prior_ce = float(-(probs[nonzero] * np.log(probs[nonzero])).sum())
    majority_exact = float(probs.max())
    best_neighbor = 0.0
    for pred_bin in range(num_bins):
        lo = max(0, pred_bin - 1)
        hi = min(num_bins, pred_bin + 2)
        best_neighbor = max(best_neighbor, float(probs[lo:hi].sum()))
    half = num_bins // 2
    neg_mass = float(probs[:half].sum())
    pos_mass = float(probs[half:].sum())
    return {
        "uniform_ce": float(math.log(num_bins)),
        "uniform_exact_accuracy": float(1.0 / num_bins),
        "uniform_neighbor_accuracy": float((3 * num_bins - 2) / (num_bins * num_bins)),
        "empirical_prior_ce": empirical_prior_ce,
        "majority_exact_accuracy": majority_exact,
        "best_constant_neighbor_accuracy": best_neighbor,
        "best_constant_direction_accuracy": max(neg_mass, pos_mass),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-dataset", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-ce-improvement", type=float, default=0.05)
    parser.add_argument("--min-exact-improvement", type=float, default=0.01)
    parser.add_argument("--min-neighbor-improvement", type=float, default=0.05)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    num_bins = int(cfg.actor.model.num_bins)
    if num_bins <= 2:
        raise ValueError(
            f"multi-bin evaluator requires num_bins > 2; got {num_bins}. "
            "Use diagnose_steam_pairs.py for the binary diagnostic."
        )
    seed = int(args.seed if args.seed is not None else cfg.data.get("seed", 42))
    # Keep runtime patch and explicit constructor seed aligned.
    import os

    os.environ["STEAM_PAIR_SEED"] = str(seed)

    entries = OmegaConf.to_container(cfg.data.train_data_paths, resolve=True)
    if not isinstance(entries, list) or not entries:
        raise ValueError("data.train_data_paths must be a non-empty list")

    datasets = [build_dataset(entry, cfg, seed) for entry in entries]
    for dataset in datasets:
        dataset.set_epoch(0)

    # Match the SFT worker's mixture-wide L_max behavior rather than evaluating
    # each leaf with a different reference percentile.
    if any(bool(ds.length_scale_enabled) for ds in datasets):
        percentile = float(cfg.data.get("length_scale_percentile", 90.0))
        reference = PairDataset.compute_global_length_scale_reference(
            datasets, percentile
        )
        for dataset in datasets:
            dataset.set_length_scale_reference(reference)
    else:
        reference = None

    device = torch.device(args.device)
    model = SteamCriticModel.from_checkpoint(
        args.checkpoint,
        device=str(device),
        tokenizer_path=str(cfg.actor.model.tokenizer_path),
        vision_repo_id=str(cfg.actor.model.vision_repo_id),
        language_repo_id=str(cfg.actor.model.language_repo_id),
        num_bins=num_bins,
        stride_k=int(cfg.data.k),
        ensemble_size=int(cfg.actor.model.get("ensemble_size", 1)),
        ensemble_head_seed_base=int(cfg.actor.model.get("ensemble_head_seed_base", 0)),
        fusion_hidden_dim=int(cfg.actor.model.get("fusion_hidden_dim", 512)),
        dropout=float(cfg.actor.model.get("dropout", 0.1)),
        max_token_len=int(cfg.actor.model.get("max_token_len", 200)),
    )
    processor = getattr(model, "processor", None)
    if processor is None:
        raise RuntimeError("loaded STEAM checkpoint has no attached processor")
    collator = BinaryPairDataCollator(
        processor=processor,
        max_length=int(cfg.actor.model.get("max_token_len", 200)),
        train=False,
        num_bins=num_bins,
    )

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    sample_counts: dict[str, int] = {}

    model.eval()
    with torch.no_grad():
        for entry, dataset in zip(entries, datasets):
            indices = fixed_indices(dataset, int(args.max_samples_per_dataset))
            sample_counts[str(entry["dataset_path"])] = len(indices)
            for start in range(0, len(indices), int(args.batch_size)):
                samples = [dataset[i] for i in indices[start : start + int(args.batch_size)]]
                batch = collator(samples)
                observation = move_to_device(batch["observation"], device)
                output = model.predict(observation)
                probs = output.probs.detach().float().cpu()
                if probs.ndim != 2 or probs.shape[1] != num_bins:
                    raise RuntimeError(
                        f"unexpected STEAM probability shape: {tuple(probs.shape)}"
                    )
                targets = batch["labels"].long().cpu()
                preds = probs.argmax(dim=-1)
                all_targets.extend(targets.tolist())
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.numpy())

    if not all_targets:
        raise RuntimeError("evaluation produced no samples")
    targets_t = torch.tensor(all_targets, dtype=torch.long)
    probs_t = torch.tensor(np.asarray(all_probs), dtype=torch.float32)
    preds_t = torch.tensor(all_preds, dtype=torch.long)
    eps = torch.finfo(probs_t.dtype).eps
    nll = F.nll_loss(torch.log(probs_t.clamp_min(eps)), targets_t).item()
    exact = (preds_t == targets_t).float().mean().item()
    neighbor = ((preds_t - targets_t).abs() <= 1).float().mean().item()
    half = num_bins // 2
    direction = ((preds_t >= half) == (targets_t >= half)).float().mean().item()
    entropy = (-(probs_t.clamp_min(eps) * torch.log(probs_t.clamp_min(eps))).sum(-1)).mean().item()

    targets_np = targets_t.numpy()
    baselines = baseline_metrics(targets_np, num_bins)
    ce_gain = baselines["empirical_prior_ce"] - nll
    exact_gain = exact - baselines["majority_exact_accuracy"]
    neighbor_gain = neighbor - baselines["best_constant_neighbor_accuracy"]

    passed = (
        ce_gain >= float(args.min_ce_improvement)
        and exact_gain >= float(args.min_exact_improvement)
        and neighbor_gain >= float(args.min_neighbor_improvement)
    )

    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "seed": seed,
        "epoch": 0,
        "num_bins": num_bins,
        "k": int(cfg.data.k),
        "length_scale_reference": reference,
        "num_samples": len(all_targets),
        "samples_per_dataset": sample_counts,
        "metrics": {
            "cross_entropy": nll,
            "exact_accuracy": exact,
            "neighbor_accuracy": neighbor,
            "direction_accuracy": direction,
            "mean_predictive_entropy": entropy,
            "max_entropy": float(math.log(num_bins)),
        },
        "baselines": baselines,
        "improvements": {
            "ce_over_empirical_prior": ce_gain,
            "exact_over_majority": exact_gain,
            "neighbor_over_best_constant": neighbor_gain,
        },
        "target_bin_histogram": {
            str(k): int(v)
            for k, v in sorted(Counter(all_targets).items())
        },
        "predicted_bin_histogram": {
            str(k): int(v)
            for k, v in sorted(Counter(all_preds).items())
        },
        "thresholds": {
            "min_ce_improvement": float(args.min_ce_improvement),
            "min_exact_improvement": float(args.min_exact_improvement),
            "min_neighbor_improvement": float(args.min_neighbor_improvement),
        },
        "pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not passed:
        print(
            "FAIL multi-bin STEAM quality gate: checkpoint does not improve "
            "enough over empirical/constant baselines."
        )
        return 6
    print("PASS multi-bin STEAM quality gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
