#!/usr/bin/env python3
"""Validate that a STEAM critic config contains only verified successful data."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from generate_xr1_steam_configs import validate_success_leaf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--skip-episode-success-check",
        action="store_true",
        help="Only validate YAML/path semantics; do not inspect is_success values.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    entries = OmegaConf.to_container(cfg.data.train_data_paths, resolve=True)
    if not isinstance(entries, list) or not entries:
        raise ValueError("data.train_data_paths must be a non-empty list")

    total_episodes = 0
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"train_data_paths[{idx}] is not a mapping")
        dataset_path = Path(str(entry.get("dataset_path", ""))).expanduser()
        if not dataset_path:
            raise ValueError(f"train_data_paths[{idx}] has no dataset_path")
        if "normal_success" not in dataset_path.parts:
            raise ValueError(
                f"train_data_paths[{idx}] is not a normal_success leaf: {dataset_path}"
            )
        if any(part.startswith("recovery_") for part in dataset_path.parts):
            raise ValueError(
                f"train_data_paths[{idx}] unexpectedly contains recovery data: {dataset_path}"
            )

        dataset_type = str(entry.get("type", "")).lower()
        only_success = bool(entry.get("only_success", False))
        if not (dataset_type == "sft" or (dataset_type == "rollout" and only_success)):
            raise ValueError(
                f"train_data_paths[{idx}] must be type=sft or rollout+only_success=true; "
                f"got type={dataset_type!r}, only_success={only_success}"
            )

        if not args.skip_episode_success_check:
            episodes, _ = validate_success_leaf(dataset_path)
            total_episodes += episodes

    model_bins = int(cfg.actor.model.num_bins)
    k = int(cfg.data.k)
    if model_bins < 2 or model_bins % 2:
        raise ValueError(f"actor.model.num_bins must be even and >=2, got {model_bins}")
    if model_bins > 2 and (2 * k) % model_bins != 0:
        raise ValueError(
            f"invalid multi-bin geometry: 2*k={2*k} is not divisible by num_bins={model_bins}"
        )

    extra = (
        f", verified_episodes={total_episodes}"
        if not args.skip_episode_success_check
        else ", episode_content_check=skipped"
    )
    print(
        f"PASS success-only config: datasets={len(entries)}, k={k}, "
        f"num_bins={model_bins}{extra}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
