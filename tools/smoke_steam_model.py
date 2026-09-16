import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.steam import BinaryPairDataCollator, PairDataset
from rlinf.models.embodiment.value_model.steam import get_model
from rlinf.models.embodiment.value_model.steam.processing import (
    SteamImageProcessor,
    SteamProcessor,
)


ROOT = "/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot"
VISION = "/data_cfs/data1/qiuguolin/models/steam/siglip-so400m-patch14-384"
LANGUAGE = "/data_cfs/data1/qiuguolin/models/steam/gemma-3-270m"
CAMERAS = ("observation.images.image", "observation.images.wrist_image")

dataset = PairDataset(
    ROOT,
    camera_keys=CAMERAS,
    k=32,
    dataset_type="sft",
    only_success=True,
    num_bins=32,
)
# LeRobot >=0.4 exposes the task table as a DataFrame. Normalize the prompt
# lookup for the current RLinf fast path.
tasks = dataset._source.meta.tasks
if hasattr(tasks, "iloc"):
    tasks = {
        int(row["task_index"]): row["task"]
        for _, row in tasks.reset_index().iterrows()
    }
dataset._source._tasks = tasks
tokenizer_processor = SteamProcessor(
    image_processor=SteamImageProcessor(
        image_size=(384, 384), image_keys=CAMERAS, do_augment=False
    ),
    tokenizer_name_or_path=LANGUAGE,
    max_token_len=200,
)
collator = BinaryPairDataCollator(
    processor=tokenizer_processor,
    max_length=200,
    train=False,
    num_bins=32,
)
batch = collator([dataset[0]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print(
    "batch image shapes:",
    {k: tuple(v.shape) for k, v in batch["observation"]["images"].items()},
)
print("batch labels:", batch["labels"].tolist())

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
print("loading SteamCriticModel...")
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
observation["tokenized_prompt_mask"] = observation["tokenized_prompt_mask"].to(device)
output = model(observation, labels=batch["labels"].to(device))
print("logits shape:", tuple(output.logits.shape))
print("loss:", float(output.loss.detach().cpu()))
output.loss.backward()
print("backward: ok")
