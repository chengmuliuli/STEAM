#!/usr/bin/env python3
"""Convert XR-1 RoboCasa LeRobot v2.1 rollout leaves to v3.0 safely.

The source tree is never modified. Each leaf is hard-linked into the user's
output tree, then the official converter rewrites only that staging copy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_info(path: Path) -> dict[str, Any]:
    with (path / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def is_v30(path: Path) -> bool:
    try:
        return read_info(path).get("codebase_version") == "v3.0"
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def leaf_name(relative: Path) -> str:
    return relative.as_posix().replace("/", "__")


def convert_one(source: Path, dest: Path, log_path: Path) -> dict[str, Any]:
    started = time.time()
    relative = source.relative_to(source.parents[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if is_v30(dest):
            return {
                "source": str(source),
                "output": str(dest),
                "status": "already_v30",
                "seconds": 0.0,
            }
        raise RuntimeError(f"Refusing to reuse non-v3 output: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Cross-user hard links are blocked on the shared filesystem.  Use a
    # regular recursive copy so the staging tree is independent and safe.
    subprocess.run(["cp", "-a", str(source), str(dest)], check=True)

    command = [
        sys.executable,
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        "--repo-id=lerobot",
        f"--root={dest.parent}",
        "--push-to-hub=false",
        "--data-file-size-in-mb=100",
        "--video-file-size-in-mb=500",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"source={source}\noutput={dest}\ncommand={' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"converter failed with exit code {result.returncode}; see {log_path}"
        )
    if not is_v30(dest):
        raise RuntimeError(f"converter finished without v3.0 metadata: {dest}")
    return {
        "source": str(source),
        "output": str(dest),
        "status": "converted",
        "seconds": round(time.time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output-root must be separate from source-root")
    output_root.mkdir(parents=True, exist_ok=True)

    leaves = sorted(
        p for p in source_root.rglob("lerobot")
        if (p / "meta" / "info.json").is_file()
    )
    if args.limit > 0:
        leaves = leaves[: args.limit]
    if not leaves:
        raise RuntimeError(f"no LeRobot v2.1 leaves found below {source_root}")

    manifest_path = output_root / "conversion_manifest.json"
    payload: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "converter": "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        "source_leaves": len(leaves),
        "completed": [],
        "failed": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_manifest(manifest_path, payload)

    print(f"Found {len(leaves)} rollout leaves", flush=True)
    for index, source in enumerate(leaves, start=1):
        relative = source.relative_to(source_root)
        dest = output_root / relative
        log_path = output_root / "_logs" / f"{index:04d}__{leaf_name(relative)}.log"
        print(f"[{index}/{len(leaves)}] {relative}", flush=True)
        try:
            record = convert_one(source, dest, log_path)
            payload["completed"].append(record)
            print(f"  {record['status']} ({record['seconds']:.1f}s)", flush=True)
        except Exception as exc:
            record = {
                "source": str(source),
                "output": str(dest),
                "status": "failed",
                "error": repr(exc),
            }
            payload["failed"].append(record)
            print(f"  FAILED: {exc}", flush=True)
        write_manifest(manifest_path, payload)

    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_manifest(manifest_path, payload)
    print(
        f"Finished: completed={len(payload['completed'])}, "
        f"failed={len(payload['failed'])}; manifest={manifest_path}",
        flush=True,
    )
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
