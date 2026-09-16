"""Runtime compatibility shims for the XR-1 STEAM overlay.

Besides the server-side PyAV decoder compatibility, this module makes STEAM
pair sampling reproducible without carrying a fork of RLinf's large SFT worker:

* multi-bin stride sampling is a pure function of
  ``(seed, epoch, dataset, pair_position)`` instead of mutable unseeded RNG
  state, so worker count / access order do not change labels;
* positive and negative logical samples for the same anchor use the same
  physical stride and are exact reversals;
* binary diagnostics can opt into ``STEAM_BINARY_STRICT_K=1`` so tail anchors
  with ``t + k >= T`` are excluded instead of clamped to a shorter stride.

The default seed comes from ``STEAM_PAIR_SEED`` (42 when unset), matching the
XR-1 config generator's default ``data.seed``.
"""

from __future__ import annotations

import os
import zlib
from typing import Any

import av
import numpy as np
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
            frame_ts = (
                float(frame.pts * stream.time_base)
                if frame.pts is not None
                else target
            )
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
    from rlinf.data.datasets.steam.binning import (
        _scaled_signed_stride_to_bin,
        _signed_stride_to_bin,
    )
    from rlinf.data.datasets.steam.pair_dataset import PairDataset, _LeRobotSource

    # ------------------------------------------------------------------
    # Existing metadata compatibility patch.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Deterministic STEAM temporal-pair sampling.
    # ------------------------------------------------------------------
    _orig_pair_init = PairDataset.__init__
    _orig_pair_len = PairDataset.__len__
    _orig_pair_getitem = PairDataset.__getitem__

    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"", "0", "false", "no", "off"}

    def deterministic_pair_init(self, *args, **kwargs):
        # Optional kwargs are consumed here so existing RLinf call sites remain
        # source-compatible even though upstream PairDataset does not expose
        # these knobs yet.
        pair_seed = kwargs.pop("seed", None)
        strict_binary_stride = kwargs.pop("strict_binary_stride", None)
        _orig_pair_init(self, *args, **kwargs)

        if pair_seed is None:
            pair_seed = int(os.environ.get("STEAM_PAIR_SEED", "42"))
        self._steam_pair_seed = int(pair_seed)
        self._steam_pair_epoch = 0
        self._steam_dataset_salt = int(
            zlib.crc32(str(self.source_name).encode("utf-8")) & 0xFFFFFFFF
        )

        if strict_binary_stride is None:
            strict_binary_stride = _env_flag("STEAM_BINARY_STRICT_K", False)
        self._steam_strict_binary_stride = bool(strict_binary_stride)
        self._steam_strict_binary_anchors: list[tuple[int, int]] | None = None

        if self._steam_strict_binary_stride:
            if int(self.num_bins) != 2:
                raise ValueError(
                    "strict_binary_stride is only valid for num_bins=2; "
                    f"got num_bins={self.num_bins}"
                )
            anchors: list[tuple[int, int]] = []
            for episode in self._eligible:
                episode_length = int(self._source.episode_length(episode))
                # Exactly t -> t+k.  No boundary clamp is allowed.
                for t in range(max(0, episode_length - int(self.k))):
                    anchors.append((int(episode), int(t)))
            if not anchors:
                raise ValueError(
                    "strict binary diagnostic found no full-k anchors; "
                    f"all eligible episodes are shorter than k+1={int(self.k)+1}"
                )
            self._steam_strict_binary_anchors = anchors

    def deterministic_pair_set_epoch(self, epoch: int) -> None:
        self._steam_pair_epoch = int(epoch)

    def deterministic_pair_len(self) -> int:
        anchors = getattr(self, "_steam_strict_binary_anchors", None)
        if anchors is not None:
            return 2 * len(anchors)
        return _orig_pair_len(self)

    def _normalise_index(idx: int, length: int) -> int:
        if idx < 0:
            idx += length
        if not 0 <= idx < length:
            raise IndexError(idx)
        return int(idx)

    def _build_from_indices(
        self,
        *,
        episode: int,
        anchor_t: int,
        frame_idx_t: int,
        frame_idx_tk: int,
        label: int,
    ) -> dict[str, Any]:
        raw_t, raw_tk = self._source.get_raw_pair(
            episode,
            frame_idx_t,
            frame_idx_tk,
            camera_keys=self.camera_keys,
        )
        # Preserve upstream behavior: language is resolved from the temporal
        # anchor even for the reversed sample.
        if frame_idx_t == anchor_t:
            prompt_sample = raw_t
        elif frame_idx_tk == anchor_t:
            prompt_sample = raw_tk
        else:
            prompt_sample = raw_t
        prompt = self._resolve_prompt_from_sample(prompt_sample, episode, anchor_t)
        return self._build_sample(
            episode=episode,
            frame_idx_t=frame_idx_t,
            frame_idx_tk=frame_idx_tk,
            prompt=prompt,
            label=int(label),
            raw_t=raw_t,
            raw_tk=raw_tk,
        )

    def deterministic_pair_getitem(self, idx: int):
        idx = _normalise_index(int(idx), len(self))
        pair_position = idx // 2
        is_positive = (idx % 2) == 0

        strict_anchors = getattr(self, "_steam_strict_binary_anchors", None)
        if strict_anchors is not None:
            episode, t = strict_anchors[pair_position]
            t2 = t + int(self.k)
            if is_positive:
                return _build_from_indices(
                    self,
                    episode=episode,
                    anchor_t=t,
                    frame_idx_t=t,
                    frame_idx_tk=t2,
                    label=1,
                )
            return _build_from_indices(
                self,
                episode=episode,
                anchor_t=t,
                frame_idx_t=t2,
                frame_idx_tk=t,
                label=0,
            )

        # Preserve upstream binary semantics unless the strict diagnostic flag
        # is enabled.  Formal STEAM uses multi-bin and therefore takes the path
        # below.
        if int(self.num_bins) == 2:
            return _orig_pair_getitem(self, idx)

        episode, t, _ = self._resolve_pair_position(pair_position)
        episode_length = int(self._source.episode_length(episode))
        max_valid_stride = min(int(self.k), episode_length - 1 - int(t))
        if max_valid_stride < 1:
            raise RuntimeError(
                f"no valid multi-bin stride for episode={episode}, t={t}, "
                f"episode_length={episode_length}"
            )

        # Stateless random draw. Positive and negative samples for the same
        # pair_position deliberately share the same draw and are exact physical
        # reversals.  The epoch changes the draw while worker scheduling and
        # access order cannot.
        seed_sequence = np.random.SeedSequence(
            [
                int(self._steam_pair_seed) & 0xFFFFFFFF,
                int(self._steam_pair_epoch) & 0xFFFFFFFF,
                int(self._steam_dataset_salt) & 0xFFFFFFFF,
                int(pair_position) & 0xFFFFFFFF,
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        stride = int(rng.integers(1, max_valid_stride + 1))
        signed_stride = stride if is_positive else -stride

        if bool(self.length_scale_enabled) and self._length_scale_reference is not None:
            scale = max(
                1.0,
                float(self._length_scale_reference) / float(episode_length),
            )
            label = _scaled_signed_stride_to_bin(
                signed_stride * scale,
                int(self.k),
                int(self.num_bins),
            )
        else:
            label = _signed_stride_to_bin(
                signed_stride,
                int(self.k),
                int(self.num_bins),
            )

        if is_positive:
            frame_idx_t, frame_idx_tk = int(t), int(t) + stride
        else:
            frame_idx_t, frame_idx_tk = int(t) + stride, int(t)
        return _build_from_indices(
            self,
            episode=int(episode),
            anchor_t=int(t),
            frame_idx_t=frame_idx_t,
            frame_idx_tk=frame_idx_tk,
            label=int(label),
        )

    PairDataset.__init__ = deterministic_pair_init
    PairDataset.set_epoch = deterministic_pair_set_epoch
    PairDataset.__len__ = deterministic_pair_len
    PairDataset.__getitem__ = deterministic_pair_getitem

except Exception as exc:
    # Keep sitecustomize import from making unrelated commands unusable, but
    # surface the reason when requested for debugging.
    if _env_flag := os.environ.get("STEAM_RUNTIME_COMPAT_VERBOSE"):
        print(f"[steam_runtime_compat] PairDataset patch skipped: {exc!r}")
