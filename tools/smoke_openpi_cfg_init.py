#!/usr/bin/env python3
"""Smoke-test the real Pi0.5 CFG model + external RoboCasa norm stats.

This intentionally calls the production ``rlinf.models.embodiment.openpi_cfg``
loader instead of mocking normalization.  A PASS therefore proves that the
configured checkpoint can be instantiated and that quantile norm stats are
actually accepted by the same loader used for CFG-RL training.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from omegaconf import OmegaConf


def _reject_placeholder(value: str, name: str) -> None:
    if not value or "/path/to/" in value:
        raise ValueError(f"{name} is still a placeholder: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.actor.model
    model_path = str(model_cfg.model_path)
    norm_stats_path = str(model_cfg.openpi_data.norm_stats_path)
    _reject_placeholder(model_path, "actor.model.model_path")
    _reject_placeholder(norm_stats_path, "actor.model.openpi_data.norm_stats_path")

    model_path_obj = Path(model_path).expanduser()
    norm_path_obj = Path(norm_stats_path).expanduser()
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Pi0.5 checkpoint does not exist: {model_path_obj}")
    if norm_path_obj.is_dir():
        norm_file = norm_path_obj / "norm_stats.json"
    else:
        norm_file = norm_path_obj
    if not norm_file.is_file():
        raise FileNotFoundError(f"norm_stats.json does not exist: {norm_file}")

    # Import only after path validation so configuration failures stay cheap.
    from rlinf.models.embodiment.openpi_cfg import get_model

    print(f"Pi0.5 checkpoint: {model_path_obj}")
    print(f"external norm stats: {norm_file}")
    print("initializing production OpenPI CFG loader ...")
    model = get_model(model_cfg)

    wrappers = getattr(model, "_input_transforms", None)
    # Wrapper internals differ between OpenPI revisions, so successful model
    # construction is the compatibility contract.  The overlay loader itself
    # already raises if quantile normalization is enabled and q01/q99 is absent.
    print(f"model class: {type(model).__module__}.{type(model).__name__}")
    if wrappers is not None:
        print("input transforms attached: yes")
    print("PASS Pi0.5 CFG initialization with external norm stats")

    del model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
