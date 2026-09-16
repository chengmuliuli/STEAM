#!/usr/bin/env python3
"""Smoke-test Pi0.5 CFG with a real XR-1 RoboCasa batch.

The test deliberately goes beyond checkpoint initialization. It takes a real
converted LeRobot episode, verifies the expected 2-view / 16D-state / 12D-action
raw contract, runs the production OpenPI input transforms and quantile
normalization, executes one real CFGRL flow-matching forward pass, and checks
that the output transform returns 12D RoboCasa actions.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

import steam_runtime_compat  # noqa: F401 -- install XR-1 video/metadata shims

from rlinf.data.datasets.steam.pair_dataset import _LeRobotSource, _to_uint8_hwc


def _reject_placeholder(value: str, name: str) -> None:
    if not value or "/path/to/" in value:
        raise ValueError(f"{name} is still a placeholder: {value!r}")


def _first_present(sample: dict[str, Any], keys: Iterable[str], name: str):
    for key in keys:
        if key in sample:
            return sample[key]
    raise KeyError(f"Could not find {name}; tried {list(keys)}; keys={sorted(sample)}")


def _as_vector(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a vector, got scalar")
    return arr.reshape(-1).astype(np.float32, copy=False)


def _to_chw_uint8(value: Any) -> np.ndarray:
    hwc = _to_uint8_hwc(value)
    return np.transpose(hwc, (2, 0, 1)).copy()


def _move(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    return value


def _find_dataset(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.rglob("lerobot")
        if (path / "meta" / "info.json").is_file()
        and "normal_success" in path.parts
    )
    if not candidates:
        candidates = sorted(
            path
            for path in root.rglob("lerobot")
            if (path / "meta" / "info.json").is_file()
        )
    if not candidates:
        raise RuntimeError(f"no converted LeRobot dataset below {root}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    norm_file = norm_path_obj / "norm_stats.json" if norm_path_obj.is_dir() else norm_path_obj
    if not norm_file.is_file():
        raise FileNotFoundError(f"norm_stats.json does not exist: {norm_file}")

    from rlinf.models.embodiment.openpi_cfg import get_model
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation

    print(f"Pi0.5 checkpoint: {model_path_obj}")
    print(f"external norm stats: {norm_file}")
    print("initializing production OpenPI CFG loader ...")
    model = get_model(model_cfg)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset_path = args.dataset.resolve() if args.dataset else _find_dataset(args.data_root.resolve())
    source = _LeRobotSource(
        str(dataset_path),
        only_success=False,
        dataset_type="rollout",
    )
    action_horizon = int(model.config.action_horizon)
    episode = next(
        (
            ep
            for ep in range(source.num_episodes())
            if source.episode_length(ep) >= action_horizon
        ),
        None,
    )
    if episode is None:
        raise RuntimeError(
            f"no episode in {dataset_path} has >= action_horizon={action_horizon} frames"
        )

    raw0 = source.get_raw_sample(episode, 0)
    state = _as_vector(
        _first_present(
            raw0,
            ("observation/state", "observation.state", "state"),
            "state",
        ),
        "state",
    )
    if state.shape[-1] != 16:
        raise ValueError(f"XR-1 smoke expects raw 16D state, got {state.shape}")

    external = _to_chw_uint8(
        _first_present(
            raw0,
            (
                "observation/image",
                "observation.image",
                "observation.images.robot0_agentview_left",
                "image_left",
            ),
            "external RGB image",
        )
    )
    wrist = _to_chw_uint8(
        _first_present(
            raw0,
            (
                "observation/wrist_image",
                "observation.wrist_image",
                "observation.images.robot0_eye_in_hand",
                "wrist_image",
            ),
            "wrist RGB image",
        )
    )

    actions: list[np.ndarray] = []
    for frame in range(action_horizon):
        raw = source.get_raw_sample(episode, frame)
        action = _as_vector(
            _first_present(raw, ("actions", "action"), "action"),
            "action",
        )
        if action.shape[-1] != 12:
            raise ValueError(
                f"XR-1 smoke expects raw 12D action, got {action.shape} at frame {frame}"
            )
        actions.append(action)
    action_chunk = np.stack(actions, axis=0)

    prompt = source.get_prompt_from_sample(raw0, episode, 0)
    if not prompt:
        prompt = source.get_prompt(episode, 0)
    if not prompt:
        raise ValueError("real XR-1 sample has no task prompt")

    # model.input_transform(transpose=True) expects batched CHW env images and
    # transposes each sample to HWC before applying RobocasaInputs.
    raw_batch = {
        "observation/state": torch.from_numpy(state[None, :]),
        "observation/image": torch.from_numpy(external[None, ...]),
        "observation/wrist_image": torch.from_numpy(wrist[None, ...]),
        "actions": torch.from_numpy(action_chunk[None, ...]),
        "prompt": [str(prompt)],
        "positive_guidance_prompt": [str(prompt)],
        "negative_guidance_prompt": [str(prompt)],
    }

    transformed = model.input_transform(raw_batch, transpose=True)
    if "state" not in transformed or "actions" not in transformed:
        raise RuntimeError(f"OpenPI transform omitted state/actions: {transformed.keys()}")
    transformed = _move(transformed, device)
    observation = Observation.from_dict(transformed)
    transformed_actions = transformed["actions"]

    # The raw 16D/12D contract may be padded internally to Pi0.5's model
    # dimensions. That is expected; assert the environment dimensions before
    # transform and the model-forward compatibility after transform.
    print(f"dataset: {dataset_path}")
    print(f"raw external image CHW: {tuple(external.shape)}")
    print(f"raw wrist image CHW: {tuple(wrist.shape)}")
    print(f"raw state: {tuple(state.shape)}")
    print(f"raw action chunk: {tuple(action_chunk.shape)}")
    print(f"transformed state: {tuple(observation.state.shape)}")
    print(f"transformed actions: {tuple(transformed_actions.shape)}")
    print(f"model action_horizon: {action_horizon}")

    if transformed_actions.shape[1] != action_horizon:
        raise ValueError(
            f"transformed action horizon mismatch: {transformed_actions.shape[1]} != {action_horizon}"
        )

    # Run the actual CFGRL forward path, including observation preprocessing,
    # guidance routing and flow-matching network execution.
    with torch.no_grad():
        loss, metrics = model(
            data={
                "observation": observation,
                "actions": transformed_actions,
                "advantage": torch.ones(1, dtype=torch.bool, device=device),
            }
        )
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Pi0.5 CFG forward returned non-finite loss: {loss}")
    print(f"real-batch CFG flow loss: {float(loss.detach().cpu()):.6f}")
    print(f"forward metric keys: {sorted(metrics.keys())}")

    # Exercise the production output transform as a shape/normalization check.
    output = model.output_transform({"actions": transformed_actions})
    output_actions = output["actions"]
    expected_chunk = int(model.config.action_chunk)
    if output_actions.ndim != 3:
        raise ValueError(f"unexpected output action rank: {tuple(output_actions.shape)}")
    if output_actions.shape[1] != expected_chunk:
        raise ValueError(
            f"output action chunk mismatch: {output_actions.shape[1]} != {expected_chunk}"
        )
    if output_actions.shape[-1] != 12:
        raise ValueError(
            f"output transform must return 12D RoboCasa actions, got {tuple(output_actions.shape)}"
        )

    print(f"output RoboCasa actions: {tuple(output_actions.shape)}")
    print("PASS Pi0.5 CFG real XR-1 batch transform + forward smoke")

    del model, source
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
