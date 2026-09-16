#!/usr/bin/env python3
"""Generate RLinf STEAM/CFG-RL configs for converted XR-1 RoboCasa rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
)
VISION = "/path/to/models/steam/siglip-so400m-patch14-384"
LANGUAGE = "/path/to/models/steam/gemma-3-270m"
PI05 = "/path/to/pi05-robocasa-human300-pytorch"


def find_v30_leaves(root: Path) -> list[Path]:
    leaves = []
    for path in sorted(root.rglob("lerobot")):
        info_path = path / "meta" / "info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("codebase_version") == "v3.0":
            leaves.append(path)
    return leaves


def dataset_entries(paths: list[Path], *, include_only_success: bool = True) -> str:
    lines = []
    for path in paths:
        lines.append(f'    - dataset_path: "{path}"')
        lines.append("      type: rollout")
        if include_only_success:
            lines.append("      only_success: false")
        lines.append("      weight: 1.0")
    return "\n".join(lines)


def value_config(paths: list[Path]) -> str:
    return f"""# Auto-generated for Xiaomi XR-1 RoboCasa365 policy rollouts.
# All entries are behavior-policy rollouts, including successful and failed episodes.
defaults:
  - model/steam_value_model@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - override hydra/job_logging: stdout
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${{oc.env:REPO_PATH}}/examples/offline_rl/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: sft
  logger:
    log_path: "/path/to/steam_xr1_v30_copy/steam_value_training"
    project_name: rlinf
    experiment_name: "steam_xr1_robocasa_value"
    logger_backends: ["tensorboard"]
  max_epochs: -1
  max_steps: 16000
  val_check_interval: -1
  save_interval: 1000

data:
  train_data_paths:
{dataset_entries(paths)}
  balance_weights: true
  seed: 42
  dataset_type: rollout
  camera_keys:
    - observation.images.robot0_agentview_left
    - observation.images.robot0_eye_in_hand
  k: 32
  only_success: false
  min_episode_length: null
  train_num_workers: 0
  eval_num_workers: 0
  prefetch_factor: null
  persistent_workers: false
  pin_memory: false
  do_augment: false

algorithm:
  adv_type: gae

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 1
  global_batch_size: 8
  seed: 0

  model:
    precision: fp32
    fusion_hidden_dim: 512
    dropout: 0.1
    label_smoothing: 0.05
    num_frames_per_pair: 2
    num_bins: 32
    vision_repo_id: "{VISION}"
    language_repo_id: "{LANGUAGE}"
    tokenizer_path: "{LANGUAGE}"
    freeze_vision_encoder: false
    freeze_language_model: false
    max_state_dim: 32
    state_discretization_bins: 256
    max_token_len: 200
    ensemble_size: 1
    ensemble_head_seed_base: 0
    use_gradient_checkpointing: true

  optim:
    lr: 5.0e-5
    value_lr: 5.0e-5
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-8
    weight_decay: 1.0e-5
    clip_grad: 10.0
    lr_scheduler: constant
    lr_warmup_steps: 500
    total_training_steps: 16000
    min_lr: 1.0e-6

  fsdp_config:
    strategy: fsdp
    sharding_strategy: no_shard
    use_orig_params: true
    gradient_checkpointing: false
    mixed_precision:
      param_dtype: bf16
      reduce_dtype: fp32
      buffer_dtype: fp32

reward:
  use_reward_model: false
critic:
  use_critic_model: false
"""


def advantage_config(paths: list[Path]) -> str:
    return f"""# Auto-generated for converted XR-1 RoboCasa365 rollouts.
advantage:
  value_checkpoint: "/path/to/steam_xr1_v30_copy/steam_value_training/steam_xr1_robocasa_value/checkpoints/global_step_16000/actor"
  batch_size: 8
  num_dataloader_workers_per_gpu: 0
  prefetch_factor: 2
  label_mode: quantile
  rollout_quantile: 0.3
  tag: steam_xr1_k32_value1
  model:
    precision: bf16

data:
  k: 32
  camera_keys:
    - observation.images.robot0_agentview_left
    - observation.images.robot0_eye_in_hand
  train_data_paths:
{dataset_entries(paths, include_only_success=False)}

distributed:
  backend: nccl
  timeout: 3600
"""


def cfg_rl_config(paths: list[Path]) -> str:
    return f"""# Auto-generated for CFG-RL with Pi05 RoboCasa365.
defaults:
  - model/pi0_5@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - _self_
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${{oc.env:REPO_PATH}}/examples/offline_rl/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: sft
  logger:
    log_path: "/path/to/steam_xr1_v30_copy/cfg_rl_training"
    project_name: rlinf
    experiment_name: "cfg_rl_xr1_robocasa"
    logger_backends: ["tensorboard"]
  max_epochs: 30000
  max_steps: -1
  val_check_interval: -1
  save_interval: 3000

data:
  num_workers: 0
  advantage_tag: steam_xr1_k32_value1
  balance_dataset_weights: true
  seed: 42
  train_data_paths:
{dataset_entries(paths, include_only_success=False)}

algorithm:
  adv_type: gae

actor:
  group_name: "ActorGroup"
  training_backend: fsdp
  micro_batch_size: 1
  global_batch_size: 8
  seed: 0

  model:
    precision: null
    model_path: "{PI05}"
    model_type: cfg_model
    add_value_head: false
    num_action_chunks: 5
    action_dim: 12
    openpi:
      config_name: pi05_robocasa365_pretrain_human300
      num_images_in_input: 2
      action_chunk: ${{actor.model.num_action_chunks}}
      action_env_dim: ${{actor.model.action_dim}}
      num_steps: 5
      train_expert_only: false
      guidance_type: positive
      unconditional_prob: 0.1
      positive_only_conditional: true
    openpi_data:
      norm_stats_path: /path/to/steam_xr1_v30_copy/robocasa_norm_stats
      state_space: 16d
      image_space: 2views
      action_space: 12d

  optim:
    lr: 1.0e-5
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-8
    weight_decay: 1.0e-10
    clip_grad: 1.0
    lr_scheduler: cosine
    lr_warmup_steps: 5000
    total_training_steps: 30000
    min_lr: 0.0

  fsdp_config:
    strategy: fsdp
    sharding_strategy: no_shard
    use_orig_params: false
    gradient_checkpointing: true
    mixed_precision:
      param_dtype: ${{actor.model.precision}}
      reduce_dtype: ${{actor.model.precision}}
      buffer_dtype: ${{actor.model.precision}}

reward:
  use_reward_model: false
critic:
  use_critic_model: false
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = find_v30_leaves(args.data_root.resolve())
    if not paths:
        raise RuntimeError(f"no converted v3.0 LeRobot leaves below {args.data_root}")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "steam_value_model_sft_robocasa_xr1.yaml": value_config(paths),
        "steam_compute_advantages_robocasa_xr1.yaml": advantage_config(paths),
        "cfg_rl_openpi_robocasa_xr1.yaml": cfg_rl_config(paths),
    }
    for name, content in outputs.items():
        (args.config_dir / name).write_text(content, encoding="utf-8")
        print(f"wrote {args.config_dir / name} ({len(paths)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
