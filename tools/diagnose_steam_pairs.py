#!/usr/bin/env python3
"""Inspect STEAM temporal pairs before training and optionally test a checkpoint.

The diagnostic deliberately uses the same PairDataset and BinaryPairDataCollator
as training.  It writes pair images + a JSON report, checks binary forward /
reverse label symmetry, reports the bin histogram, and (when --checkpoint is
supplied) verifies that a trained binary critic flips its preference when the
same physical pair is reversed.
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
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from rlinf.data.datasets.steam import BinaryPairDataCollator, PairDataset
from rlinf.models.embodiment.value_model.steam.modeling_critic import SteamCriticModel


def move_to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def build_dataset(entry: dict, cfg) -> PairDataset:
    return PairDataset(
        dataset_path=str(entry["dataset_path"]),
        camera_keys=tuple(cfg.data.camera_keys),
        k=int(cfg.data.k),
        dataset_type=str(entry.get("type", cfg.data.get("dataset_type", "rollout"))),
        only_success=bool(entry.get("only_success", cfg.data.get("only_success", False))),
        min_episode_length=cfg.data.get("min_episode_length", None),
        num_bins=int(cfg.actor.model.num_bins),
        length_scale_enabled=bool(cfg.data.get("length_scale_enabled", False)),
        length_scale_percentile=float(cfg.data.get("length_scale_percentile", 90.0)),
    )


def save_pair_image(sample: dict, out_path: Path) -> None:
    panels: list[Image.Image] = []
    for camera in sorted(set(sample["image_t"]) | set(sample["image_tk"])):
        left = sample["image_t"].get(camera)
        right = sample["image_tk"].get(camera)
        if left is None or right is None:
            continue
        left_img = Image.fromarray(np.asarray(left, dtype=np.uint8))
        right_img = Image.fromarray(np.asarray(right, dtype=np.uint8))
        h = max(left_img.height, right_img.height)
        panel = Image.new("RGB", (left_img.width + right_img.width, h + 28))
        panel.paste(left_img, (0, 28))
        panel.paste(right_img, (left_img.width, 28))
        draw = ImageDraw.Draw(panel)
        draw.text((4, 6), f"{camera}: t", fill="white")
        draw.text((left_img.width + 4, 6), "t2", fill="white")
        panels.append(panel)
    if not panels:
        return
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels)
    canvas = Image.new("RGB", (width, height))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def mean_abs_image_diff(sample: dict) -> tuple[float | None, bool]:
    diffs: list[float] = []
    all_equal = True
    common = sorted(set(sample["image_t"]) & set(sample["image_tk"]))
    for camera in common:
        a = sample["image_t"].get(camera)
        b = sample["image_tk"].get(camera)
        if a is None or b is None:
            continue
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.shape != b.shape:
            all_equal = False
            continue
        diffs.append(float(np.mean(np.abs(a - b))))
        all_equal = all_equal and bool(np.array_equal(a, b))
    return (float(np.mean(diffs)) if diffs else None), all_equal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=64)
    parser.add_argument("--save-pairs", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-prediction-pairs", type=int, default=32)
    parser.add_argument("--pass-loss-threshold", type=float, default=0.65)
    parser.add_argument("--pass-accuracy-threshold", type=float, default=0.65)
    parser.add_argument("--pass-flip-threshold", type=float, default=0.65)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    entries = OmegaConf.to_container(cfg.data.train_data_paths, resolve=True)
    if not isinstance(entries, list) or not entries:
        raise ValueError("data.train_data_paths must be a non-empty list")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = Counter()
    duplicate_pairs = 0
    sampled = 0
    records: list[dict[str, Any]] = []
    physical_binary_pairs: list[tuple[dict, dict]] = []
    datasets: list[PairDataset] = []

    for ds_idx, entry in enumerate(entries):
        dataset = build_dataset(entry, cfg)
        datasets.append(dataset)
        n = min(len(dataset), max(2, args.samples_per_dataset))
        # Start at zero so binary logical indices are inspected as exact
        # (+ direction, - direction) pairs for each temporal anchor.
        for index in range(n):
            sample = dataset[index]
            label = int(sample["label"])
            labels[label] += 1
            raw_stride = int(sample["frame_idx_tk"]) - int(sample["frame_idx_t"])
            mad, exact_duplicate = mean_abs_image_diff(sample)
            duplicate_pairs += int(exact_duplicate)
            sampled += 1
            record = {
                "dataset_index": ds_idx,
                "dataset_path": str(entry["dataset_path"]),
                "logical_index": index,
                "episode": int(sample["episode"]),
                "frame_idx_t": int(sample["frame_idx_t"]),
                "frame_idx_t2": int(sample["frame_idx_tk"]),
                "raw_signed_stride": raw_stride,
                "label": label,
                "mean_abs_pixel_diff": mad,
                "exact_duplicate_all_common_cameras": exact_duplicate,
                "prompt": sample["prompt"],
            }
            records.append(record)
            if len(records) <= args.save_pairs:
                save_pair_image(
                    sample,
                    args.output_dir / "pairs" / f"pair_{len(records)-1:04d}.png",
                )

        if int(cfg.actor.model.num_bins) == 2:
            for pair_pos in range(min(dataset.num_pair_positions, n // 2)):
                pos = dataset[2 * pair_pos]
                neg = dataset[2 * pair_pos + 1]
                symmetric = (
                    int(pos["episode"]) == int(neg["episode"])
                    and int(pos["frame_idx_t"]) == int(neg["frame_idx_tk"])
                    and int(pos["frame_idx_tk"]) == int(neg["frame_idx_t"])
                    and int(pos["label"]) == 1
                    and int(neg["label"]) == 0
                )
                if not symmetric:
                    raise AssertionError(
                        "Binary PairDataset forward/reverse symmetry failed at "
                        f"dataset={entry['dataset_path']} pair_position={pair_pos}: "
                        f"pos=({pos['frame_idx_t']},{pos['frame_idx_tk']},label={pos['label']}), "
                        f"neg=({neg['frame_idx_t']},{neg['frame_idx_tk']},label={neg['label']})"
                    )
                physical_binary_pairs.append((pos, neg))

    report: dict[str, Any] = {
        "config": str(args.config),
        "num_bins": int(cfg.actor.model.num_bins),
        "k": int(cfg.data.k),
        "num_sampled": sampled,
        "label_histogram": {str(k): int(v) for k, v in sorted(labels.items())},
        "exact_duplicate_pairs": duplicate_pairs,
        "exact_duplicate_fraction": (duplicate_pairs / sampled if sampled else None),
        "records": records,
    }

    if sampled == 0:
        raise RuntimeError("No pairs were sampled")
    if duplicate_pairs == sampled:
        raise RuntimeError(
            "Every sampled pair is pixel-identical across all common cameras; "
            "video timestamp/decode alignment is likely broken."
        )
    if int(cfg.actor.model.num_bins) == 2:
        if set(labels) != {0, 1}:
            raise RuntimeError(f"Binary diagnostic did not observe both labels: {labels}")
        imbalance = abs(labels[1] - labels[0]) / max(1, labels[1] + labels[0])
        if imbalance > 0.05:
            raise RuntimeError(f"Binary labels are unexpectedly imbalanced: {labels}")

    prediction_pass = None
    if args.checkpoint is not None:
        if int(cfg.actor.model.num_bins) != 2:
            raise ValueError("--checkpoint prediction-flip diagnostic currently requires num_bins=2")
        device = torch.device(args.device)
        model = SteamCriticModel.from_checkpoint(
            args.checkpoint,
            device=str(device),
            tokenizer_path=str(cfg.actor.model.tokenizer_path),
            vision_repo_id=str(cfg.actor.model.vision_repo_id),
            language_repo_id=str(cfg.actor.model.language_repo_id),
            num_bins=2,
            stride_k=int(cfg.data.k),
            ensemble_size=1,
            precision="fp32",
            fusion_hidden_dim=int(cfg.actor.model.fusion_hidden_dim),
            dropout=float(cfg.actor.model.dropout),
            max_token_len=int(cfg.actor.model.max_token_len),
        )
        collator = BinaryPairDataCollator(
            processor=model.processor,
            max_length=int(cfg.actor.model.max_token_len),
            train=False,
            num_bins=2,
        )
        max_pairs = min(args.max_prediction_pairs, len(physical_binary_pairs))
        if max_pairs < 1:
            raise RuntimeError("No physical binary pair available for checkpoint diagnostic")

        correct = 0
        flips = 0
        losses: list[float] = []
        prediction_records: list[dict[str, Any]] = []
        for pos, neg in physical_binary_pairs[:max_pairs]:
            batch = collator([pos, neg])
            observation = move_to_device(batch["observation"], device)
            target = batch["labels"].to(device)
            with torch.inference_mode():
                output = model(observation, labels=target)
            probs = output.probs.detach().float().cpu()
            pred = probs.argmax(dim=-1)
            correct += int((pred == batch["labels"]).sum().item())
            p_progress_forward = float(probs[0, 1])
            p_progress_reverse = float(probs[1, 1])
            flips += int(p_progress_forward > 0.5 and p_progress_reverse < 0.5)
            sample_loss = torch.nn.functional.cross_entropy(
                output.logits.detach().float().cpu(), batch["labels"], reduction="none"
            )
            losses.extend(float(v) for v in sample_loss)
            prediction_records.append(
                {
                    "episode": int(pos["episode"]),
                    "forward": [int(pos["frame_idx_t"]), int(pos["frame_idx_tk"])],
                    "p_progress_forward": p_progress_forward,
                    "p_progress_reverse": p_progress_reverse,
                }
            )

        accuracy = correct / float(2 * max_pairs)
        flip_rate = flips / float(max_pairs)
        mean_loss = float(np.mean(losses))
        prediction_pass = (
            mean_loss < args.pass_loss_threshold
            and accuracy >= args.pass_accuracy_threshold
            and flip_rate >= args.pass_flip_threshold
        )
        report["checkpoint_diagnostic"] = {
            "checkpoint": str(args.checkpoint),
            "pairs": max_pairs,
            "mean_cross_entropy": mean_loss,
            "random_binary_baseline": math.log(2.0),
            "accuracy": accuracy,
            "flip_rate": flip_rate,
            "thresholds": {
                "loss_lt": args.pass_loss_threshold,
                "accuracy_ge": args.pass_accuracy_threshold,
                "flip_rate_ge": args.pass_flip_threshold,
            },
            "pass": prediction_pass,
            "predictions": prediction_records,
        }

    report_path = args.output_dir / "pair_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pair diagnostic report: {report_path}")
    print(f"label histogram: {dict(sorted(labels.items()))}")
    print(f"exact duplicate fraction: {duplicate_pairs}/{sampled} = {duplicate_pairs/sampled:.4f}")
    if args.checkpoint is not None:
        diag = report["checkpoint_diagnostic"]
        status = "PASS" if prediction_pass else "FAIL"
        print(
            f"{status} checkpoint diagnostic: loss={diag['mean_cross_entropy']:.4f} "
            f"(random={math.log(2.0):.4f}), acc={diag['accuracy']:.3f}, "
            f"flip_rate={diag['flip_rate']:.3f}"
        )
        return 0 if prediction_pass else 3
    print("PASS pair/label diagnostic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
