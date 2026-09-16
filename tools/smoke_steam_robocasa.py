#!/usr/bin/env python3
"""Forward/backward smoke test for STEAM on a converted XR-1 RoboCasa leaf."""
import os

import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.steam import BinaryPairDataCollator, PairDataset
from rlinf.models.embodiment.value_model.steam import get_model
from rlinf.models.embodiment.value_model.steam.processing import (
    SteamImageProcessor,
    SteamProcessor,
)


ROOT = os.environ.get(
    "STEAM_ROBOCASA_ROOT",
    "/data_cfs/data1/qiuguolin/steam_xr1_v30_test/"
    "CloseBlenderLid/gpu6_worker0/normal_success/CloseBlenderLid/lerobot",
)
VISION = "/data_cfs/data1/qiuguolin/models/steam/siglip-so400m-patch14-384"
LANGUAGE = "/data_cfs/data1/qiuguolin/models/steam/gemma-3-270m"
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
processor = SteamProcessor(
    image_processor=SteamImageProcessor(
        image_size=(384, 384), image_keys=CAMERAS, do_augment=False
    ),
    tokenizer_name_or_path=LANGUAGE,
    max_token_len=200,
)
collator = BinaryPairDataCollator(
    processor=processor,
    max_length=200,
    train=True,
    num_bins=32,
)
batch = collator([dataset[0]])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
print(
    "dataset:", ROOT,
    "episodes:", dataset._source.num_episodes(),
    "pairs:", len(dataset),
    flush=True,
)
print(
    "image shapes:",
    {k: tuple(v.shape) for k, v in batch["observation"]["images"].items()},
    flush=True,
)

cfg = OmegaConf.create(
    {
        "vision_repo_id": VISION,
        "language_repo_id": LANGUAGE,
        "precision": "fp32",
        "fusion_hidden_dim": 512,
        "dropout": 0.1,
        "label_smoothing": 0.05,
        "num_frames_per_pair": 2,
        "num_bins": 32,
        "stride_k": 32,
        "ensemble_size": 1,
        "freeze_vision_encoder": True,
        "freeze_language_model": True,
        "use_gradient_checkpointing": False,
        "max_token_len": 200,
    }
)
print("loading STEAM value model...", flush=True)
model = get_model(cfg).to(device)
model.train()
observation = batch["observation"]
observation["images"] = {
    k: v.to(device) for k, v in observation["images"].items()
}
observation["image_masks"] = {
    k: v.to(device) for k, v in observation["image_masks"].items()
}
observation["tokenized_prompt"] = observation["tokenized_prompt"].to(device)
observation["tokenized_prompt_mask"] = observation[
    "tokenized_prompt_mask"
].to(device)
output = model(observation, labels=batch["labels"].to(device))
print("logits shape:", tuple(output.logits.shape), flush=True)
print("loss:", float(output.loss.detach().cpu()), flush=True)
output.loss.backward()
print("backward: ok", flush=True)
