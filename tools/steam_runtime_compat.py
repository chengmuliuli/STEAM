"""Runtime shims for the server's FFmpeg-free STEAM smoke run."""

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

try:
    from rlinf.data.datasets.steam.pair_dataset import _LeRobotSource

    def metadata_sample_with_dataframe_tasks(self, episode, frame):
        global_idx = self._ep_starts[episode] + int(frame)
        raw_dataset = self.base.hf_dataset
        sample = {
            key: raw_dataset.data.column(key)[global_idx].as_py()
            for key in raw_dataset.column_names
        }
        task_idx = sample.get("task_index")
        if task_idx is not None:
            task_idx = int(task_idx)
            tasks = self.meta.tasks
            if hasattr(tasks, "iloc"):
                row = tasks.iloc[task_idx]
                sample["task"] = row["task"] if "task" in row.index else row.iloc[0]
            else:
                sample["task"] = tasks[task_idx]
        return sample

    _LeRobotSource._metadata_sample = metadata_sample_with_dataframe_tasks
except Exception:
    pass
