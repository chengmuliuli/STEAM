#!/usr/bin/env python3
"""Instantiate every converted XR-1 leaf and verify STEAM metadata contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlinf.data.datasets.steam import PairDataset


CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    leaves = []
    for path in sorted(args.data_root.resolve().rglob("lerobot")):
        info_path = path / "meta" / "info.json"
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("codebase_version") == "v3.0":
            leaves.append(path)
    episodes = 0
    pairs = 0
    for index, path in enumerate(leaves, start=1):
        dataset = PairDataset(
            path,
            camera_keys=CAMERAS,
            k=32,
            dataset_type="rollout",
            only_success=False,
            num_bins=32,
        )
        if len(dataset) <= 0:
            raise RuntimeError(f"empty PairDataset: {path}")
        episodes += dataset._source.num_episodes()
        pairs += len(dataset)
        if index == 1 or index == len(leaves) or index % 20 == 0:
            print(
                f"[{index}/{len(leaves)}] episodes={episodes} pairs={pairs}",
                flush=True,
            )
    print(
        f"verified leaves={len(leaves)} episodes={episodes} pairs={pairs}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
