#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:?Set REPO_PATH to the RLinf checkout}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the converted rollout workspace}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
NUM_BINS="${NUM_BINS:-8}"
DIAGNOSTIC_STEPS="${DIAGNOSTIC_STEPS:-2048}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
CONFIG_DIR="${REPO_PATH}/examples/offline_rl/config"
SUFFIX="_diag_b${NUM_BINS}"
CONFIG_NAME="steam_value_model_sft_robocasa_xr1${SUFFIX}"
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
EXPERIMENT_NAME="steam_xr1_success_diag_b${NUM_BINS}"
DIAGNOSTIC_DIR="${RUN_ROOT}/steam_diagnostics/${CONFIG_NAME}"

: "${STEAM_VISION_MODEL:?Set STEAM_VISION_MODEL to the local SigLIP checkpoint}"
: "${STEAM_LANGUAGE_MODEL:?Set STEAM_LANGUAGE_MODEL to the local Gemma checkpoint}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if (( NUM_BINS <= 2 || NUM_BINS % 2 != 0 || 64 % NUM_BINS != 0 )); then
    echo "ERROR: NUM_BINS must be an even divisor of 64 and >2; got ${NUM_BINS}" >&2
    exit 2
fi

mkdir -p "${DIAGNOSTIC_DIR}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_PATH}/tools:${REPO_PATH}:${PYTHONPATH:-}"
export HF_HOME="${RUN_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${RUN_ROOT}/hf_datasets_cache"
export TRANSFORMERS_CACHE="${RUN_ROOT}/hf_cache/transformers"
export TMPDIR="${RUN_ROOT}/tmp"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off
export STEAM_PAIR_SEED="${STEAM_PAIR_SEED:-42}"
unset STEAM_BINARY_STRICT_K || true

cd "${REPO_PATH}"

# Intentionally keep length scaling OFF in the 8-bin intermediate stage. This
# isolates temporal-distance resolution from episode-length normalization.
"${PYTHON_BIN}" tools/generate_xr1_steam_configs.py \
    --data-root "${DATA_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --config-dir "${CONFIG_DIR}" \
    --config-suffix "${SUFFIX}" \
    --success-only \
    --num-bins "${NUM_BINS}" \
    --ensemble-size 1 \
    --value-max-steps "${DIAGNOSTIC_STEPS}" \
    --value-save-interval "${DIAGNOSTIC_STEPS}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --value-experiment-name "${EXPERIMENT_NAME}" \
    --vision-model "${STEAM_VISION_MODEL}" \
    --language-model "${STEAM_LANGUAGE_MODEL}"

"${PYTHON_BIN}" tools/validate_steam_success_config.py --config "${CONFIG_PATH}"
"${PYTHON_BIN}" tools/diagnose_steam_pairs.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${DIAGNOSTIC_DIR}/pretrain" \
    --samples-per-dataset 128 \
    --save-pairs 8

gpu_info="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
free_gpu="$(printf '%s\n' "${gpu_info}" | awk -F, -v min="${MIN_FREE_MIB}" '$2+0 >= min {gsub(/ /, "", $1); print $1; exit}')"
if [[ ! "${free_gpu}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: no GPU has at least ${MIN_FREE_MIB} MiB free" >&2
    printf '%s\n' "${gpu_info}" >&2
    exit 5
fi
export CUDA_VISIBLE_DEVICES="${free_gpu}"

echo "$(date '+%F %T') starting ${NUM_BINS}-bin intermediate critic diagnostic"
echo "Python=${PYTHON_BIN}; seed=${STEAM_PAIR_SEED}; GPU=${free_gpu}"

# Freeze the large backbones here as well: this stage asks whether fixed visual
# features contain enough information to resolve coarse temporal distance.
"${PYTHON_BIN}" examples/offline_rl/advantage_labeling/steam/train_steam.py \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    actor.model.ensemble_size=1 \
    actor.model.freeze_vision_encoder=true \
    actor.model.freeze_language_model=true \
    actor.model.label_smoothing=0.0 \
    actor.optim.lr_warmup_steps=0 \
    runner.max_steps="${DIAGNOSTIC_STEPS}" \
    runner.save_interval="${DIAGNOSTIC_STEPS}" \
    actor.optim.total_training_steps="${DIAGNOSTIC_STEPS}"

CHECKPOINT_PATH="${RUN_ROOT}/steam_value_training/${EXPERIMENT_NAME}/checkpoints/global_step_${DIAGNOSTIC_STEPS}/actor"
if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: expected ${NUM_BINS}-bin checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 4
fi

"${PYTHON_BIN}" tools/evaluate_steam_multibin_checkpoint.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${DIAGNOSTIC_DIR}/posttrain_metrics.json" \
    --max-samples-per-dataset 256 \
    --device cuda

echo "$(date '+%F %T') PASS ${NUM_BINS}-bin intermediate STEAM diagnostic"
