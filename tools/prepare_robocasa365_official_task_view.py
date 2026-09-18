import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE = Path(os.environ.get("STEAM_SOURCE", "/data-cfs/data1/datasets/robocasa365-pretrain-mg"))
TARGET = Path(
    os.environ.get(
        "STEAM_TARGET",
        "/data-cfs/data1/qiuguolin/robocasa365-pretrain-mg-official-steam-task",
    )
)
DROP_CAMERA = "observation.images.robot0_agentview_left"
ROUND_TIMESTAMPS = os.environ.get("STEAM_ROUND_TIMESTAMPS", "0") == "1"

if TARGET.exists():
    raise SystemExit(f"refusing to overwrite existing target: {TARGET}")

tasks_df = pd.read_parquet(SOURCE / "meta" / "tasks.parquet")
if "task_index" not in tasks_df.columns:
    raise SystemExit("source tasks.parquet has no task_index column")
if tasks_df.index.name is None:
    raise SystemExit("source tasks.parquet has no task-text index")
task_mapping = {
    int(row.task_index): str(task_text)
    for task_text, row in tasks_df.iterrows()
}
if sorted(task_mapping) != list(range(len(task_mapping))):
    raise SystemExit("task_index values are not contiguous from zero")

TARGET.mkdir(parents=True)
(TARGET / "videos").symlink_to(SOURCE / "videos", target_is_directory=True)

meta_target = TARGET / "meta"
meta_target.mkdir()
(meta_target / "episodes").symlink_to(SOURCE / "meta" / "episodes", target_is_directory=True)
(meta_target / "stats.json").symlink_to(SOURCE / "meta" / "stats.json")

info = json.loads((SOURCE / "meta" / "info.json").read_text())
fps = float(info.get("fps", 20))
features = info["features"]
features.pop(DROP_CAMERA, None)
features.pop("task_index", None)
features["task"] = {
    "dtype": "string",
    "shape": [1],
    "fps": fps,
}
(meta_target / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

# Make the task sidecar explicit and easy for both LeRobot and RLinf to read.
tasks_out = tasks_df.reset_index().rename(columns={tasks_df.index.name: "task"})
tasks_out.to_parquet(meta_target / "tasks.parquet", index=False)

data_target = TARGET / "data"
data_target.mkdir()
source_paths = sorted((SOURCE / "data").glob("*/*.parquet"))
if not source_paths:
    raise SystemExit("source dataset has no data parquet files")

for position, source_path in enumerate(source_paths, start=1):
    rel = source_path.relative_to(SOURCE / "data")
    destination = data_target / rel
    destination.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(source_path)
    if "task_index" not in table.column_names:
        raise SystemExit(f"missing task_index in {source_path}")
    if ROUND_TIMESTAMPS and "timestamp" in table.column_names:
        timestamp_index = table.schema.get_field_index("timestamp")
        timestamps = np.asarray(table["timestamp"].combine_chunks())
        timestamps = (np.rint(timestamps * fps) / fps).astype(timestamps.dtype)
        table = table.set_column(
            timestamp_index,
            "timestamp",
            pa.array(timestamps, type=table["timestamp"].type),
        )
    task_values = [task_mapping[int(value)] for value in table["task_index"].to_pylist()]
    table = table.drop(["task_index"])
    table = table.append_column("task", pa.array(task_values, type=pa.string()))
    pq.write_table(table, destination, compression="zstd", use_dictionary=True)
    if position == 1 or position % 20 == 0 or position == len(source_paths):
        print(f"converted {position}/{len(source_paths)}: {rel}", flush=True)

print("created", TARGET)
print("source_episodes", info["total_episodes"])
print("source_frames", info["total_frames"])
print("tasks", len(task_mapping))
print("video_features", [k for k, v in features.items() if v.get("dtype") == "video"])
print("timestamps_rounded_to_fps", ROUND_TIMESTAMPS, fps)
print("source_unchanged", SOURCE)

