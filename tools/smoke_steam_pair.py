from pathlib import Path

import av
import torch

import lerobot.datasets.lerobot_dataset as lerobot_dataset_module


def decode_video_frames_pyav_direct(video_path, timestamps, tolerance_s, backend=None):
    del tolerance_s, backend
    output = []
    for timestamp in timestamps:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        target = float(timestamp)
        seek_pts = int(target / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        best = None
        best_distance = float("inf")
        for frame in container.decode(stream):
            frame_ts = float(frame.pts * stream.time_base) if frame.pts is not None else target
            distance = abs(frame_ts - target)
            if distance < best_distance:
                best = frame
                best_distance = distance
            if frame_ts >= target and best is not None:
                break
        container.close()
        if best is None:
            raise RuntimeError(f"No frame decoded from {video_path} at {target}")
        output.append(torch.from_numpy(best.to_ndarray(format="rgb24")))
    return torch.stack(output)


lerobot_dataset_module.decode_video_frames = decode_video_frames_pyav_direct

from rlinf.data.datasets.steam.pair_dataset import PairDataset


ROOT = "/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot"
dataset = PairDataset(
    ROOT,
    camera_keys=("observation.images.image", "observation.images.wrist_image"),
    k=4,
    dataset_type="sft",
    only_success=True,
)
# LeRobot 0.4.4 exposes tasks as a DataFrame; RLinf's current metadata fast
# path expects the older dict shape. Normalize it for this smoke test.
tasks = dataset._source.meta.tasks
if hasattr(tasks, "iloc"):
    dataset._source.meta.tasks = {
        int(row["task_index"]): row["task"]
        for _, row in tasks.reset_index().iterrows()
    }
print("pair length:", len(dataset))
sample = dataset[0]
print("sample keys:", sorted(sample.keys()))
print("prompt:", sample["prompt"])
for key in ("image_t", "image_tk", "image_mask_t", "image_mask_tk"):
    print(key, {cam: (getattr(value, "shape", None), getattr(value, "dtype", None)) if hasattr(value, "shape") else value for cam, value in sample[key].items()})
print("labels:", sample["label"], "episode:", sample["episode"], "frames:", sample["frame_idx_t"], sample["frame_idx_tk"])
