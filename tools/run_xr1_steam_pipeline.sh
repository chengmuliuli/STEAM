#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:?Set REPO_PATH to the RLinf checkout containing this overlay}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the converted XR-1 rollout workspace}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}}"
CONFIG_PATH="${REPO_PATH}/examples/offline_rl/config"
LOG_ROOT="${RUN_ROOT}/pipeline"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
FORMAL_ENSEMBLE_SIZE="${FORMAL_ENSEMBLE_SIZE:-3}"
FORMAL_MICRO_BATCH_SIZE="${FORMAL_MICRO_BATCH_SIZE:-8}"
FORMAL_GLOBAL_BATCH_SIZE="${FORMAL_GLOBAL_BATCH_SIZE:-64}"

: "${STEAM_VISION_MODEL:?Set STEAM_VISION_MODEL to the local SigLIP checkpoint}"
: "${STEAM_LANGUAGE_MODEL:?Set STEAM_LANGUAGE_MODEL to the local Gemma checkpoint}"
: "${PI05_MODEL:?Set PI05_MODEL to the local Pi0.5 checkpoint}"
: "${ROBOCASA_NORM_STATS:?Set ROBOCASA_NORM_STATS to norm_stats.json or its directory}"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/pipeline.log") 2>&1

export PYTHONPATH="${REPO_PATH}/.venv/lib/python3.10/site-packages:${REPO_PATH}/tools:${REPO_PATH}:${PYTHONPATH:-}"
export REPO_PATH RUN_ROOT
export HF_HOME="${RUN_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${RUN_ROOT}/hf_datasets_cache"
export TRANSFORMERS_CACHE="${RUN_ROOT}/hf_cache/transformers"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off
export RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:47999}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/qgl-ray-steam}"
mkdir -p "${RAY_TMPDIR}"

cd "${REPO_PATH}"
echo "$(date '+%F %T') XR-1 STEAM pipeline starting."
echo "repo_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "data_root=${DATA_ROOT}"
echo "run_root=${RUN_ROOT}"
echo "formal_ensemble_size=${FORMAL_ENSEMBLE_SIZE}"

# Generate all three configs from one invocation so value checkpoint, advantage
# tag, length scaling and CFG advantage_tag cannot drift apart.
"${PYTHON_BIN}" tools/generate_xr1_steam_configs.py \
    --data-root "${DATA_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --config-dir "${CONFIG_PATH}" \
    --success-only \
    --num-bins 32 \
    --length-scale-enabled \
    --length-scale-percentile 90 \
    --ensemble-size "${FORMAL_ENSEMBLE_SIZE}" \
    --micro-batch-size "${FORMAL_MICRO_BATCH_SIZE}" \
    --global-batch-size "${FORMAL_GLOBAL_BATCH_SIZE}" \
    --value-max-steps 16000 \
    --value-save-interval 1000 \
    --value-experiment-name steam_xr1_robocasa_value \
    --cfg-experiment-name cfg_rl_xr1_robocasa \
    --vision-model "${STEAM_VISION_MODEL}" \
    --language-model "${STEAM_LANGUAGE_MODEL}" \
    --pi05-model "${PI05_MODEL}" \
    --norm-stats-path "${ROBOCASA_NORM_STATS}"

VALUE_CONFIG="${CONFIG_PATH}/steam_value_model_sft_robocasa_xr1.yaml"
ADV_CONFIG="${CONFIG_PATH}/steam_compute_advantages_robocasa_xr1.yaml"
CFG_CONFIG="${CONFIG_PATH}/cfg_rl_openpi_robocasa_xr1.yaml"

"${PYTHON_BIN}" tools/validate_steam_success_config.py --config "${VALUE_CONFIG}"
"${PYTHON_BIN}" tools/diagnose_steam_pairs.py \
    --config "${VALUE_CONFIG}" \
    --output-dir "${RUN_ROOT}/steam_diagnostics/formal_pretrain" \
    --samples-per-dataset 64 \
    --save-pairs 12

# Optional GPU pinning. If PIPELINE_CUDA_VISIBLE_DEVICES is unset we leave GPU
# visibility untouched so the user's Ray/distributed setup can use multiple GPUs.
if [[ -n "${PIPELINE_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${PIPELINE_CUDA_VISIBLE_DEVICES}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Fail fast if no visible GPU has a reasonable amount of free memory. This is
# only a preflight; actual multi-GPU placement is handled by the RLinf launcher.
gpu_info="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
if ! printf '%s\n' "${gpu_info}" | awk -F, -v min="${MIN_FREE_MIB}" '$2+0 >= min {found=1} END {exit !found}'; then
    echo "ERROR: no GPU currently has at least ${MIN_FREE_MIB} MiB free" >&2
    printf '%s\n' "${gpu_info}" >&2
    exit 5
fi

echo "$(date '+%F %T') starting STEAM value SFT."
"${PYTHON_BIN}" examples/offline_rl/advantage_labeling/steam/train_steam.py \
    --config-path "${CONFIG_PATH}" \
    --config-name steam_value_model_sft_robocasa_xr1
echo "$(date '+%F %T') value SFT finished."

echo "$(date '+%F %T') starting STEAM advantage computation."
"${PYTHON_BIN}" examples/offline_rl/advantage_labeling/steam/process/compute_advantages_ensemble.py \
    --config-path "${CONFIG_PATH}" \
    --config-name steam_compute_advantages_robocasa_xr1
echo "$(date '+%F %T') advantage computation finished."

# Exercise the exact production CFG loader with the actual Pi0.5 checkpoint and
# external norm stats before a long policy-training job is launched.
echo "$(date '+%F %T') smoke-testing Pi0.5 CFG initialization."
"${PYTHON_BIN}" tools/smoke_openpi_cfg_init.py --config "${CFG_CONFIG}"

echo "$(date '+%F %T') starting Pi0.5 CFG-RL."
"${PYTHON_BIN}" examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py \
    --config-path "${CONFIG_PATH}" \
    --config-name cfg_rl_openpi_robocasa_xr1
echo "$(date '+%F %T') Pi0.5 CFG-RL finished."
