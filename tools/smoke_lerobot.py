from pathlib import Path
import inspect
import av
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
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


ROOT = Path("/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot")

print("constructor:", inspect.signature(LeRobotDataset))
print("getitem:", inspect.signature(LeRobotDataset.__getitem__))
dataset = LeRobotDataset(
    repo_id="libero_10_no_noops_lerobot", root=ROOT, video_backend="pyav"
)
print("length:", len(dataset))
for idx in (0, len(dataset) // 2, len(dataset) - 1):
    sample = dataset[idx]
    print("sample index:", idx, "episode:", int(sample["episode_index"]))
    print("keys:", sorted(sample.keys()))
    for key in ("observation.images.image", "observation.images.wrist_image"):
        value = sample[key]
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
