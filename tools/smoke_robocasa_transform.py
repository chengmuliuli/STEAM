#!/usr/bin/env python3
"""CPU-only smoke test for the RoboCasa OpenPi data transform."""
import numpy as np

from rlinf.data.datasets.steam import PairDataset
from rlinf.models.embodiment.openpi.policies.robocasa_policy import RobocasaInputs


ROOT = (
    "/data_cfs/data1/qiuguolin/steam_xr1_v30_test/"
    "CloseBlenderLid/gpu6_worker0/normal_success/CloseBlenderLid/lerobot"
)
CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
)

dataset = PairDataset(
    ROOT,
    camera_keys=CAMERAS,
    k=32,
    dataset_type="rollout",
    only_success=False,
    num_bins=32,
)
raw = dataset._source.get_raw_sample(0, 0)
print("raw keys:", sorted(raw.keys()), flush=True)
if "observation/state" not in raw and "observation.state" in raw:
    raw["observation/state"] = raw["observation.state"]
if "actions" not in raw and "action" in raw:
    raw["actions"] = np.asarray(raw["action"])[None, ...]
if "observation/image" not in raw:
    raw["observation/image"] = raw[CAMERAS[0]]
if "observation/wrist_image" not in raw:
    raw["observation/wrist_image"] = raw[CAMERAS[1]]

transform = RobocasaInputs(
    state_space="16d",
    image_space="2views",
    action_space="12d",
    model_type=None,
)
out = transform(raw)
print("state shape:", np.asarray(out["state"]).shape, flush=True)
print("actions shape:", np.asarray(out["actions"]).shape, flush=True)
print(
    "image shapes:",
    {k: np.asarray(v).shape for k, v in out["image"].items()},
    "masks:",
    {k: bool(v) for k, v in out["image_mask"].items()},
    flush=True,
)
