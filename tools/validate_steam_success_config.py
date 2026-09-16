#!/usr/bin/env python3
"""Validate that a STEAM critic config contains only verified XR-1 success data."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from generate_xr1_steam_configs import (
    SUCCESS_STATUSES,
    _success_status,
    _task_name_from_success_path,
    validate_success_leaf,
)


def _check_expected(label: str, actual: int, expected: int) -> None:
    if expected > 0 and actual != expected:
        raise ValueError(f"expected {expected} {label}, got {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--skip-episode-success-check",
        action="store_true",
        help=(
            "Only validate YAML/path semantics; do not inspect the XR-1 final "
            "official-success annotation. Not recommended for formal runs."
        ),
    )
    parser.add_argument("--expect-success-leaves", type=int, default=0)
    parser.add_argument("--expect-success-episodes", type=int, default=0)
    parser.add_argument("--expect-success-tasks", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    entries = OmegaConf.to_container(cfg.data.train_data_paths, resolve=True)
    if not isinstance(entries, list) or not entries:
        raise ValueError("data.train_data_paths must be a non-empty list")

    total_episodes = 0
    task_names: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"train_data_paths[{idx}] is not a mapping")
        raw_path = str(entry.get("dataset_path", "")).strip()
        if not raw_path:
            raise ValueError(f"train_data_paths[{idx}] has no dataset_path")
        dataset_path = Path(raw_path).expanduser()

        try:
            status = _success_status(dataset_path)
        except ValueError as exc:
            raise ValueError(
                f"train_data_paths[{idx}] is not an XR-1 success leaf; expected "
                f"one of {SUCCESS_STATUSES}: {dataset_path}"
            ) from exc
        task_names.add(_task_name_from_success_path(dataset_path))

        # Once an XR-1 leaf has been externally verified from the original
        # final-current-official-success annotation, PairDataset should consume
        # it as SFT. Do not route it through rollout+only_success, which would
        # reintroduce the unrelated `is_success` schema requirement.
        dataset_type = str(entry.get("type", "")).lower()
        only_success = bool(entry.get("only_success", False))
        if dataset_type != "sft":
            raise ValueError(
                f"train_data_paths[{idx}] ({status}) must use type=sft after "
                f"external success validation, got type={dataset_type!r}"
            )
        if not only_success:
            raise ValueError(
                f"train_data_paths[{idx}] ({status}) must retain only_success=true "
                "as an explicit config-level semantic assertion"
            )

        if not args.skip_episode_success_check:
            if not dataset_path.exists():
                raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")
            episodes, _task_name = validate_success_leaf(dataset_path)
            total_episodes += episodes

    _check_expected("success leaves", len(entries), args.expect_success_leaves)
    _check_expected("tasks", len(task_names), args.expect_success_tasks)
    if args.skip_episode_success_check:
        if args.expect_success_episodes > 0:
            raise ValueError(
                "cannot enforce --expect-success-episodes while "
                "--skip-episode-success-check is active"
            )
    else:
        _check_expected(
            "successful episodes", total_episodes, args.expect_success_episodes
        )

    model_bins = int(cfg.actor.model.num_bins)
    k = int(cfg.data.k)
    if model_bins < 2 or model_bins % 2:
        raise ValueError(f"actor.model.num_bins must be even and >=2, got {model_bins}")
    if model_bins > 2 and (2 * k) % model_bins != 0:
        raise ValueError(
            f"invalid multi-bin geometry: 2*k={2*k} is not divisible by num_bins={model_bins}"
        )

    print(f"PASS success-only config: datasets={len(entries)}")
    print(f"tasks={len(task_names)}: {', '.join(sorted(task_names))}")
    if args.skip_episode_success_check:
        print("episode_content_check=skipped")
    else:
        print(f"validated_successful_episodes={total_episodes}")
    print(f"k={k}, num_bins={model_bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
