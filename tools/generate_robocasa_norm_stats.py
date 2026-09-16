#!/usr/bin/env python3
"""Aggregate LeRobot state/action stats into an OpenPi norm_stats asset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(root: Path, key: str) -> dict:
    total = 0.0
    sum_x = None
    sum_x2 = None
    minimum = None
    maximum = None
    leaves = 0
    for stats_path in sorted(root.rglob("meta/stats.json")):
        info_path = stats_path.parent / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("codebase_version") != "v3.0" or key not in stats:
            continue
        item = stats[key]
        mean = np.asarray(item["mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(item["std"], dtype=np.float64).reshape(-1)
        count = float(np.asarray(item["count"], dtype=np.float64).reshape(-1)[0])
        lo = np.asarray(item["min"], dtype=np.float64).reshape(-1)
        hi = np.asarray(item["max"], dtype=np.float64).reshape(-1)
        if count <= 0:
            continue
        if sum_x is None:
            sum_x = np.zeros_like(mean)
            sum_x2 = np.zeros_like(mean)
            minimum = lo.copy()
            maximum = hi.copy()
        if mean.shape != sum_x.shape:
            raise ValueError(f"inconsistent {key} dimensions in {stats_path}")
        total += count
        sum_x += count * mean
        sum_x2 += count * (std * std + mean * mean)
        minimum = np.minimum(minimum, lo)
        maximum = np.maximum(maximum, hi)
        leaves += 1
    if total <= 0 or sum_x is None:
        raise RuntimeError(f"no v3.0 stats found for {key} below {root}")
    mean = sum_x / total
    variance = np.maximum(sum_x2 / total - mean * mean, 0.0)
    std = np.sqrt(variance)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        # LeRobot v3 stores min/max rather than quantiles.  They provide a
        # conservative valid fallback if the Pi05 transform requests q01/q99.
        "q01": minimum.tolist(),
        "q99": maximum.tolist(),
        "_count": int(total),
        "_leaves": leaves,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    state = aggregate(root, "observation.state")
    actions = aggregate(root, "action")
    if len(state["mean"]) != 16:
        raise ValueError(f"expected 16d state, got {len(state['mean'])}")
    if len(actions["mean"]) != 12:
        raise ValueError(f"expected 12d action, got {len(actions['mean'])}")
    count = {
        "state_samples": state.pop("_count"),
        "state_leaves": state.pop("_leaves"),
        "action_samples": actions.pop("_count"),
        "action_leaves": actions.pop("_leaves"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"norm_stats": {"state": state, "actions": actions}}
    out_path = args.output_dir / "norm_stats.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(count, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
