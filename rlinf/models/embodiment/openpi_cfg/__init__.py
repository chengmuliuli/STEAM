# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
import pathlib

import torch
from omegaconf import DictConfig


def _load_external_norm_stats(norm_stats_path, checkpoints_module):
    """Load OpenPI norm stats from either a directory or norm_stats.json path.

    OpenPI's checkpoint helper expects ``load_norm_stats(parent, asset_id)`` and
    then reads ``parent / asset_id / norm_stats.json``.  Convert an explicit
    file/directory path into that pair so CFG training honors
    ``actor.model.openpi_data.norm_stats_path`` just like the regular OpenPI
    loader does.
    """
    path = pathlib.Path(str(norm_stats_path)).expanduser()
    if path.is_file():
        if path.name != "norm_stats.json":
            raise ValueError(
                "openpi_data.norm_stats_path must point to norm_stats.json or "
                f"its containing directory, got file: {path}"
            )
        stats_dir = path.parent
    else:
        stats_dir = path

    stats_file = stats_dir / "norm_stats.json"
    if not stats_file.is_file():
        raise FileNotFoundError(
            "Could not find OpenPI norm stats at "
            f"{stats_file}. Set actor.model.openpi_data.norm_stats_path to "
            "either the file or its containing directory."
        )

    return checkpoints_module.load_norm_stats(stats_dir.parent, stats_dir.name)


def get_model(cfg: DictConfig, torch_dtype=None):
    del torch_dtype

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )

    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name,
        model_path=cfg.model_path,
        data_kwargs=data_kwargs,
    )
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForCFGActionPrediction = OpenPi0ForCFGActionPrediction(
        actor_model_config
    )
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        # Prefer stats already materialized by the data config; otherwise load
        # exactly the external asset requested by the YAML.
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_stats = _load_external_norm_stats(norm_stats_path, _checkpoints)
    else:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    if norm_stats is None:
        raise ValueError(
            "No normalization statistics were loaded for CFG training. "
            "Set actor.model.openpi_data.norm_stats_path or provide checkpoint assets."
        )

    # Fail early with an actionable message for Pi0.5, which uses quantile
    # normalization by default.
    if data_config.use_quantile_norm:
        missing_quantiles = [
            key
            for key, stats in norm_stats.items()
            if getattr(stats, "q01", None) is None or getattr(stats, "q99", None) is None
        ]
        if missing_quantiles:
            raise ValueError(
                "Quantile normalization is enabled but q01/q99 are missing for: "
                + ", ".join(sorted(missing_quantiles))
            )

    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
