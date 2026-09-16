#!/usr/bin/env python3
"""Generate reproducible RLinf STEAM/CFG-RL configs for XR-1 RoboCasa rollouts.

XR-1 success semantics are intentionally validated outside PairDataset:

* critic supervision comes from ``normal_success`` and ``recovery_success``;
* an episode is successful iff the *last frame* has
  ``annotation.recovery.final_current_official_success == True``;
* after validation these leaves are emitted as ``type: sft`` so PairDataset
  does not require a synthetic ``is_success`` column;
* advantage / CFG data may come from a separate full-rollout root containing
  both successful and failed behavior-policy trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_VISION = "/path/to/models/steam/siglip-so400m-patch14-384"
DEFAULT_LANGUAGE = "/path/to/models/steam/gemma-3-270m"
DEFAULT_PI05 = "/path/to/pi05-robocasa-human300-pytorch"
K = 32
CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
)
SUCCESS_STATUSES = ("normal_success", "recovery_success")
SUCCESS_FIELD = "annotation.recovery.final_current_official_success"


def find_v30_leaves(root: Path) -> list[Path]:
    leaves: list[Path] = []
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


def _success_status(path: Path) -> str:
    matches = [status for status in SUCCESS_STATUSES if status in path.parts]
    if len(matches) != 1:
        raise ValueError(
            f"success critic leaf must contain exactly one of {SUCCESS_STATUSES}, "
            f"got {matches!r}: {path}"
        )
    return matches[0]


def _success_paths(paths: list[Path]) -> list[Path]:
    return [
        path
        for path in paths
        if any(status in path.parts for status in SUCCESS_STATUSES)
    ]


def _task_name_from_success_path(path: Path) -> str:
    """Return the task directory immediately above the success-status folder."""
    status = _success_status(path)
    status_idx = path.parts.index(status)
    if status_idx == 0:
        raise ValueError(f"cannot infer task name above status folder in {path}")
    return path.parts[status_idx - 1]


def validate_success_leaf(path: Path) -> tuple[int, str]:
    """Verify every episode in a success leaf using the XR-1 official label.

    ``SUCCESS_FIELD`` is a per-frame *current success state*. Earlier frames are
    therefore allowed to be false; only the episode's final frame must be true.
    The function deliberately does not require or synthesize an ``is_success``
    column.
    """
    status = _success_status(path)

    try:
        from rlinf.data.datasets.steam.pair_dataset import _LeRobotSource
    except ImportError as exc:
        raise RuntimeError(
            "Strict success validation requires running this generator from the "
            "RLinf checkout with the STEAM overlay on PYTHONPATH."
        ) from exc

    source = _LeRobotSource(str(path), only_success=False, dataset_type="rollout")
    raw_dataset = source.base.hf_dataset
    if SUCCESS_FIELD not in raw_dataset.column_names:
        raise ValueError(
            f"{path} has no {SUCCESS_FIELD!r} column. Refusing to mark it as "
            "type=sft because success cannot be verified from the original XR-1 "
            "annotation. Do not use a synthetic is_success column as a substitute."
        )

    column = raw_dataset.data.column(SUCCESS_FIELD)
    bad: list[int] = []
    for episode, start in enumerate(source._ep_starts):
        length = int(source.episode_length(episode))
        if length < 1:
            bad.append(episode)
            continue
        last_row = int(start) + length - 1
        if not source._coerce_success_flag(column[last_row].as_py()):
            bad.append(episode)

    if bad:
        preview = ", ".join(str(v) for v in bad[:16])
        more = " ..." if len(bad) > 16 else ""
        raise ValueError(
            f"{path} ({status}) contains {len(bad)} episode(s) whose final "
            f"{SUCCESS_FIELD} is false: {preview}{more}"
        )
    return source.num_episodes(), _task_name_from_success_path(path)


def validate_success_paths(paths: list[Path]) -> dict[str, object]:
    total_episodes = 0
    tasks: set[str] = set()
    for path in paths:
        episodes, task_name = validate_success_leaf(path)
        total_episodes += episodes
        tasks.add(task_name)
    return {
        "leaves": len(paths),
        "episodes": total_episodes,
        "tasks": len(tasks),
        "task_names": sorted(tasks),
    }


def _enforce_expected(label: str, actual: int, expected: int) -> None:
    if expected > 0 and actual != expected:
        raise RuntimeError(
            f"success dataset count mismatch for {label}: expected {expected}, got {actual}"
        )


def dataset_entries(
    paths: list[Path],
    *,
    dataset_type: str = "rollout",
    include_only_success: bool = True,
    only_success: bool = False,
) -> str:
    lines: list[str] = []
    for path in paths:
        lines.append(f'    - dataset_path: "{path}"')
        lines.append(f"      type: {dataset_type}")
        if include_only_success:
            lines.append(f"      only_success: {'true' if only_success else 'false'}")
        lines.append("      weight: 1.0")
    return "\n".join(lines)


def _auto_tag(
    *,
    success_only: bool,
    num_bins: int,
    length_scale_enabled: bool,
    length_scale_percentile: float,
    ensemble_size: int,
) -> str:
    source = "succ" if success_only else "mixed"
    scale = f"ls{length_scale_percentile:g}" if length_scale_enabled else "nols"
    return f"steam_xr1_{source}_b{num_bins}_k{K}_{scale}_e{ensemble_size}"


def value_config(
    paths: list[Path],
    *,
    run_root: Path,
    experiment_name: str,
    success_only: bool,
    num_bins: int,
    length_scale_enabled: bool,
    length_scale_percentile: float,
    ensemble_size: int,
    max_steps: int,
    save_interval: int,
    micro_batch_size: int,
    global_batch_size: int,
    vision_model: str,
    language_model: str,
) -> str:
    dataset_type = "sft" if success_only else "rollout"
    success_flag = "true" if success_only else "false"
    length_flag = "true" if length_scale_enabled else "false"
    source_comment = (
        "# Critic uses validated normal_success + recovery_success trajectories as expert-like temporal supervision."
        if success_only
        else "# Critic uses all behavior-policy rollouts, including failed trajectories."
    )
    return f"""# Auto-generated by tools/generate_xr1_steam_configs.py.
{source_comment}
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
    log_path: "{run_root / 'steam_value_training'}"
    project_name: rlinf
    experiment_name: "{experiment_name}"
    logger_backends: ["tensorboard"]
  max_epochs: -1
  max_steps: {max_steps}
  val_check_interval: -1
  save_interval: {save_interval}
data:
  train_data_paths:
{dataset_entries(paths, dataset_type=dataset_type, only_success=success_only)}
  balance_weights: true
  seed: 42
  dataset_type: {dataset_type}
  camera_keys:
    - {CAMERAS[0]}
    - {CAMERAS[1]}
  k: {K}
  only_success: {success_flag}
  length_scale_enabled: {length_flag}
  length_scale_percentile: {length_scale_percentile:g}
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
  micro_batch_size: {micro_batch_size}
  global_batch_size: {global_batch_size}
  seed: 0

  model:
    precision: fp32
    fusion_hidden_dim: 512
    dropout: 0.1
    label_smoothing: 0.05
    num_frames_per_pair: 2
    num_bins: {num_bins}
    vision_repo_id: "{vision_model}"
    language_repo_id: "{language_model}"
    tokenizer_path: "{language_model}"
    freeze_vision_encoder: false
    freeze_language_model: false
    max_state_dim: 32
    state_discretization_bins: 256
    max_token_len: 200
    ensemble_size: {ensemble_size}
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
    total_training_steps: {max_steps}
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


def advantage_config(
    paths: list[Path],
    *,
    value_checkpoint: str,
    advantage_tag: str,
    length_scale_enabled: bool,
    length_scale_percentile: float,
) -> str:
    length_flag = "true" if length_scale_enabled else "false"
    return f"""# Auto-generated by tools/generate_xr1_steam_configs.py.
advantage:
  value_checkpoint: "{value_checkpoint}"
  batch_size: 8
  num_dataloader_workers_per_gpu: 0
  prefetch_factor: 2
  label_mode: quantile
  rollout_quantile: 0.3
  tag: "{advantage_tag}"
  model:
    precision: bf16

data:
  k: {K}
  length_scale_enabled: {length_flag}
  length_scale_percentile: {length_scale_percentile:g}
  camera_keys:
    - {CAMERAS[0]}
    - {CAMERAS[1]}
  train_data_paths:
{dataset_entries(paths, include_only_success=False)}

distributed:
  backend: nccl
  timeout: 3600
"""


def cfg_rl_config(
    paths: list[Path],
    *,
    run_root: Path,
    experiment_name: str,
    advantage_tag: str,
    pi05_model: str,
    norm_stats_path: str,
) -> str:
    return f"""# Auto-generated by tools/generate_xr1_steam_configs.py.
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
    log_path: "{run_root / 'cfg_rl_training'}"
    project_name: rlinf
    experiment_name: "{experiment_name}"
    logger_backends: ["tensorboard"]
  max_epochs: 30000
  max_steps: -1
  val_check_interval: -1
  save_interval: 3000

data:
  num_workers: 0
  advantage_tag: "{advantage_tag}"
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
    model_path: "{pi05_model}"
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
      norm_stats_path: "{norm_stats_path}"
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
    # Backward-compatible single-root option. New formal runs should use the
    # two explicit roots below.
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--critic-data-root", type=Path, default=None)
    parser.add_argument("--rollout-data-root", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--config-suffix",
        default="",
        help="Suffix added to all generated config stems, e.g. _diag_b2",
    )
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--skip-success-content-validation", action="store_true")
    parser.add_argument("--expect-success-leaves", type=int, default=0)
    parser.add_argument("--expect-success-episodes", type=int, default=0)
    parser.add_argument("--expect-success-tasks", type=int, default=0)
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--length-scale-enabled", action="store_true")
    parser.add_argument("--length-scale-percentile", type=float, default=90.0)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--value-max-steps", type=int, default=16000)
    parser.add_argument("--value-save-interval", type=int, default=1000)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--value-experiment-name", default="steam_xr1_robocasa_value")
    parser.add_argument("--cfg-experiment-name", default="cfg_rl_xr1_robocasa")
    parser.add_argument("--value-checkpoint", default=None)
    parser.add_argument("--advantage-tag", default=None)
    parser.add_argument("--vision-model", default=DEFAULT_VISION)
    parser.add_argument("--language-model", default=DEFAULT_LANGUAGE)
    parser.add_argument("--pi05-model", default=DEFAULT_PI05)
    parser.add_argument("--norm-stats-path", default=None)
    args = parser.parse_args()

    if args.num_bins < 2 or args.num_bins % 2 != 0:
        raise ValueError("--num-bins must be even and >= 2")
    if args.num_bins > 2 and (2 * K) % args.num_bins != 0:
        raise ValueError(
            f"For k={K}, 2*k={2*K} must be divisible by num_bins; got {args.num_bins}"
        )
    if not 0.0 < args.length_scale_percentile <= 100.0:
        raise ValueError("--length-scale-percentile must be in (0, 100]")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be >= 1")
    if args.value_max_steps < 1:
        raise ValueError("--value-max-steps must be >= 1")
    if args.config_suffix and not args.config_suffix.startswith("_"):
        raise ValueError(
            "--config-suffix must be empty or start with '_' (example: _diag_b2)"
        )

    critic_root_arg = args.critic_data_root or args.data_root
    if critic_root_arg is None:
        raise ValueError("set --critic-data-root (or legacy --data-root)")
    rollout_root_arg = args.rollout_data_root or args.data_root or critic_root_arg

    critic_data_root = critic_root_arg.resolve()
    rollout_data_root = rollout_root_arg.resolve()
    run_root = (args.run_root or rollout_data_root).resolve()

    critic_all_paths = find_v30_leaves(critic_data_root)
    if not critic_all_paths:
        raise RuntimeError(
            f"no converted v3.0 LeRobot leaves below critic root {critic_data_root}"
        )
    rollout_paths = find_v30_leaves(rollout_data_root)
    if not rollout_paths:
        raise RuntimeError(
            f"no converted v3.0 LeRobot leaves below rollout root {rollout_data_root}"
        )

    critic_paths = critic_all_paths
    success_summary: dict[str, object] | None = None
    if args.success_only:
        critic_paths = _success_paths(critic_all_paths)
        if not critic_paths:
            raise RuntimeError(
                "--success-only was requested, but no normal_success or "
                "recovery_success datasets were found"
            )

        task_names = sorted({_task_name_from_success_path(path) for path in critic_paths})
        if args.skip_success_content_validation:
            if args.expect_success_episodes > 0:
                raise ValueError(
                    "cannot enforce --expect-success-episodes while "
                    "--skip-success-content-validation is active"
                )
            success_summary = {
                "leaves": len(critic_paths),
                "episodes": None,
                "tasks": len(task_names),
                "task_names": task_names,
            }
            print(
                "WARNING: success content validation skipped; this is not "
                "recommended for formal training."
            )
        else:
            success_summary = validate_success_paths(critic_paths)

        _enforce_expected(
            "leaves", int(success_summary["leaves"]), args.expect_success_leaves
        )
        _enforce_expected(
            "tasks", int(success_summary["tasks"]), args.expect_success_tasks
        )
        if success_summary["episodes"] is not None:
            _enforce_expected(
                "episodes",
                int(success_summary["episodes"]),
                args.expect_success_episodes,
            )

        print(f"critic datasets: {success_summary['leaves']}")
        if success_summary["episodes"] is not None:
            print(f"validated successful episodes: {success_summary['episodes']}")
        print(f"tasks: {success_summary['tasks']}")
        print("task names: " + ", ".join(success_summary["task_names"]))

    advantage_tag = args.advantage_tag or _auto_tag(
        success_only=args.success_only,
        num_bins=args.num_bins,
        length_scale_enabled=args.length_scale_enabled,
        length_scale_percentile=args.length_scale_percentile,
        ensemble_size=args.ensemble_size,
    )
    value_checkpoint = args.value_checkpoint or str(
        run_root
        / "steam_value_training"
        / args.value_experiment_name
        / "checkpoints"
        / f"global_step_{args.value_max_steps}"
        / "actor"
    )
    norm_stats_path = args.norm_stats_path or str(run_root / "robocasa_norm_stats")

    suffix = args.config_suffix
    args.config_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"steam_value_model_sft_robocasa_xr1{suffix}.yaml": value_config(
            critic_paths,
            run_root=run_root,
            experiment_name=args.value_experiment_name,
            success_only=args.success_only,
            num_bins=args.num_bins,
            length_scale_enabled=args.length_scale_enabled,
            length_scale_percentile=args.length_scale_percentile,
            ensemble_size=args.ensemble_size,
            max_steps=args.value_max_steps,
            save_interval=args.value_save_interval,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            vision_model=args.vision_model,
            language_model=args.language_model,
        ),
        f"steam_compute_advantages_robocasa_xr1{suffix}.yaml": advantage_config(
            rollout_paths,
            value_checkpoint=value_checkpoint,
            advantage_tag=advantage_tag,
            length_scale_enabled=args.length_scale_enabled,
            length_scale_percentile=args.length_scale_percentile,
        ),
        f"cfg_rl_openpi_robocasa_xr1{suffix}.yaml": cfg_rl_config(
            rollout_paths,
            run_root=run_root,
            experiment_name=args.cfg_experiment_name,
            advantage_tag=advantage_tag,
            pi05_model=args.pi05_model,
            norm_stats_path=norm_stats_path,
        ),
    }
    for name, content in outputs.items():
        path = args.config_dir / name
        path.write_text(content, encoding="utf-8")
        count = len(critic_paths) if name.startswith("steam_value_model") else len(rollout_paths)
        print(f"wrote {path} ({count} datasets)")

    print(
        "generated settings: "
        f"suffix={suffix!r}, success_only={args.success_only}, num_bins={args.num_bins}, "
        f"length_scale={args.length_scale_enabled}, ensemble={args.ensemble_size}, "
        f"critic_root={critic_data_root}, rollout_root={rollout_data_root}, "
        f"value_checkpoint={value_checkpoint}, advantage_tag={advantage_tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
