#!/usr/bin/env bash
set -o pipefail

export PYTHONPATH="/data_cfs/data1/qiuguolin/RLinf/tools:/data_cfs/data1/qiuguolin/RLinf:${PYTHONPATH:-}"
export PATH="/data_cfs/data1/qiuguolin/RLinf/.venv/bin:${PATH}"
export TMPDIR=/data_cfs/data1/qiuguolin/tmp
export HF_HOME=/data_cfs/data1/qiuguolin/hf_cache
export HF_DATASETS_CACHE=/data_cfs/data1/qiuguolin/hf_datasets_cache
export TRANSFORMERS_CACHE=/data_cfs/data1/qiuguolin/hf_cache/transformers
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off

cd /data_cfs/data1/qiuguolin/RLinf
bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
  steam_value_model_sft \
  runner.max_steps=1 \
  runner.save_interval=1 \
  data.train_data_paths='[{dataset_path:/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot,type:sft}]' \
  data.camera_keys='[observation.images.image,observation.images.wrist_image]' \
  data.k=1 \
  data.train_num_workers=0 \
  data.eval_num_workers=0 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=1 \
  actor.optim.total_training_steps=1 \
  actor.model.vision_repo_id=/data_cfs/data1/qiuguolin/models/steam/siglip-so400m-patch14-384 \
  actor.model.language_repo_id=/data_cfs/data1/qiuguolin/models/steam/gemma-3-270m \
  actor.model.tokenizer_path=/data_cfs/data1/qiuguolin/models/steam/gemma-3-270m \
  actor.model.freeze_vision_encoder=true \
  actor.model.freeze_language_model=true \
  actor.model.use_gradient_checkpointing=false \
  actor.fsdp_config.sharding_strategy=no_shard \
  actor.fsdp_config.mixed_precision.param_dtype=bf16 \
  actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  actor.fsdp_config.mixed_precision.buffer_dtype=fp32
