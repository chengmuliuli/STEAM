#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:?Set REPO_PATH to the RLinf checkout}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the converted rollout workspace}"
CRITIC_CONFIG="${CRITIC_CONFIG:-steam_value_model_sft_robocasa_xr1_diag_b2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
DIAGNOSTIC_STEPS="${DIAGNOSTIC_STEPS:-512}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-${RUN_ROOT}/steam_diagnostics/${CRITIC_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
CONFIG_DIR="${REPO_PATH}/examples/offline_rl/config"
LOG_ROOT="${RUN_ROOT}/steam_value_training/queued_success_critic"
mkdir -p "${LOG_ROOT}" "${DIAGNOSTIC_DIR}"
exec > >(tee -a "${LOG_ROOT}/queue.log") 2>&1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN explicitly if this checkout uses another environment." >&2
    exit 2
fi

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

# Reproducible temporal-pair sampling. Binary diagnostics are deliberately
# strict: every sample is exactly +/-k frames, never a clamped short tail pair.
export STEAM_PAIR_SEED="${STEAM_PAIR_SEED:-42}"
export STEAM_BINARY_STRICT_K=1

CONFIG_PATH="${CONFIG_DIR}/${CRITIC_CONFIG}.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: binary diagnostic config not found: ${CONFIG_PATH}" >&2
    echo "Generate it with:" >&2
    echo "  ${PYTHON_BIN} tools/generate_xr1_steam_configs.py \\" >&2
    echo "    --data-root \"${RUN_ROOT}\" --run-root \"${RUN_ROOT}\" \\" >&2
    echo "    --config-dir \"${CONFIG_DIR}\" --config-suffix _diag_b2 \\" >&2
    echo "    --success-only --num-bins 2 --value-max-steps ${DIAGNOSTIC_STEPS} \\" >&2
    echo "    --value-save-interval ${DIAGNOSTIC_STEPS} \\" >&2
    echo "    --value-experiment-name steam_xr1_success_diag_b2" >&2
    exit 2
fi

cd "${REPO_PATH}"
"${PYTHON_BIN}" tools/validate_steam_success_config.py --config "${CONFIG_PATH}"

CONFIG_NUM_BINS="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(int(cfg.actor.model.num_bins))
PY
)"
if [[ "${CONFIG_NUM_BINS}" != "2" ]]; then
    echo "ERROR: ${CONFIG_PATH} has num_bins=${CONFIG_NUM_BINS}; this diagnostic requires 2 bins." >&2
    exit 2
fi

"${PYTHON_BIN}" tools/diagnose_steam_pairs.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${DIAGNOSTIC_DIR}/pretrain" \
    --samples-per-dataset 64 \
    --save-pairs 12

echo "$(date '+%F %T') queued success-only fixed-k binary critic diagnostic"
echo "Python: ${PYTHON_BIN}"
echo "STEAM_PAIR_SEED=${STEAM_PAIR_SEED}; strict_binary_k=${STEAM_BINARY_STRICT_K}"
echo "Waiting for any GPU to have at least ${MIN_FREE_MIB} MiB free."
while true; do
    gpu_info="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    free_gpu="$(printf '%s\n' "${gpu_info}" | awk -F, -v min="${MIN_FREE_MIB}" '$2+0 >= min {gsub(/ /, "", $1); print $1; exit}')"
    if [[ "${free_gpu}" =~ ^[0-9]+$ ]]; then
        free_mib="$(printf '%s\n' "${gpu_info}" | awk -F, -v id="${free_gpu}" '$1+0 == id {gsub(/ /, "", $2); print $2; exit}')"
        export CUDA_VISIBLE_DEVICES="${free_gpu}"
        echo "$(date '+%F %T') GPU ${free_gpu} available: ${free_mib} MiB free"
        break
    fi
    echo "$(date '+%F %T') all GPUs busy; retry in 60s"
    sleep 60
done

readarray -t CFG_FIELDS < <("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(cfg.runner.logger.log_path)
print(cfg.runner.logger.experiment_name)
PY
)
VALUE_LOG_PATH="${CFG_FIELDS[0]}"
VALUE_EXPERIMENT_NAME="${CFG_FIELDS[1]}"
CHECKPOINT_PATH="${VALUE_LOG_PATH}/${VALUE_EXPERIMENT_NAME}/checkpoints/global_step_${DIAGNOSTIC_STEPS}/actor"

echo "$(date '+%F %T') starting 2-bin fixed-k overfit diagnostic with ${CRITIC_CONFIG}"

# Call train_steam.py directly. The upstream run_steam_sft.sh unconditionally
# overrides runner.logger.log_path, which makes checkpoint discovery depend on
# a timestamped shell log directory and breaks reproducible automation.
"${PYTHON_BIN}" examples/offline_rl/advantage_labeling/steam/train_steam.py \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CRITIC_CONFIG}" \
    actor.model.num_bins=2 \
    actor.model.ensemble_size=1 \
    actor.model.freeze_vision_encoder=true \
    actor.model.freeze_language_model=true \
    actor.model.label_smoothing=0.0 \
    actor.micro_batch_size=8 \
    actor.global_batch_size=8 \
    actor.optim.lr_warmup_steps=0 \
    runner.max_steps="${DIAGNOSTIC_STEPS}" \
    runner.save_interval="${DIAGNOSTIC_STEPS}" \
    actor.optim.total_training_steps="${DIAGNOSTIC_STEPS}"

if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: expected diagnostic checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 4
fi

"${PYTHON_BIN}" tools/diagnose_steam_pairs.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${DIAGNOSTIC_DIR}/posttrain" \
    --samples-per-dataset 64 \
    --save-pairs 4 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --device cuda

echo "$(date '+%F %T') PASS success-only fixed-k 2-bin critic diagnostic"
