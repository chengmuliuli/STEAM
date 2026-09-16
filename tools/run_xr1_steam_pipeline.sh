#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:?Set REPO_PATH to the RLinf checkout containing this overlay}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the converted XR-1 rollout workspace}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}}"
CONFIG_PATH="${REPO_PATH}/examples/offline_rl/config"
LOG_ROOT="${RUN_ROOT}/pipeline"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
FORMAL_ENSEMBLE_SIZE="${FORMAL_ENSEMBLE_SIZE:-3}"
FORMAL_MICRO_BATCH_SIZE="${FORMAL_MICRO_BATCH_SIZE:-8}"
FORMAL_GLOBAL_BATCH_SIZE="${FORMAL_GLOBAL_BATCH_SIZE:-64}"
FORMAL_STEPS="${FORMAL_STEPS:-16000}"
RUN_BINARY_DIAGNOSTIC="${RUN_BINARY_DIAGNOSTIC:-1}"
RUN_8BIN_DIAGNOSTIC="${RUN_8BIN_DIAGNOSTIC:-1}"
BINARY_DIAGNOSTIC_STEPS="${BINARY_DIAGNOSTIC_STEPS:-512}"
EIGHT_BIN_DIAGNOSTIC_STEPS="${EIGHT_BIN_DIAGNOSTIC_STEPS:-2048}"
FORMAL_MIN_CE_IMPROVEMENT="${FORMAL_MIN_CE_IMPROVEMENT:-0.05}"
FORMAL_MIN_EXACT_IMPROVEMENT="${FORMAL_MIN_EXACT_IMPROVEMENT:-0.01}"
FORMAL_MIN_NEIGHBOR_IMPROVEMENT="${FORMAL_MIN_NEIGHBOR_IMPROVEMENT:-0.05}"

: "${STEAM_VISION_MODEL:?Set STEAM_VISION_MODEL to the local SigLIP checkpoint}"
: "${STEAM_LANGUAGE_MODEL:?Set STEAM_LANGUAGE_MODEL to the local Gemma checkpoint}"
: "${PI05_MODEL:?Set PI05_MODEL to the local Pi0.5 checkpoint}"
: "${ROBOCASA_NORM_STATS:?Set ROBOCASA_NORM_STATS to norm_stats.json or its directory}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
    echo "Expected the RLinf virtualenv by default; set PYTHON_BIN explicitly if needed." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/pipeline.log") 2>&1

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_PATH}/tools:${REPO_PATH}:${PYTHONPATH:-}"
export REPO_PATH RUN_ROOT DATA_ROOT
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
echo "python=${PYTHON_BIN}"
echo "repo_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "data_root=${DATA_ROOT}"
echo "run_root=${RUN_ROOT}"
echo "formal_ensemble_size=${FORMAL_ENSEMBLE_SIZE}"

# ---------------------------------------------------------------------------
# Stage A: fixed-k binary direction overfit diagnostic.
# ---------------------------------------------------------------------------
if [[ "${RUN_BINARY_DIAGNOSTIC}" != "0" ]]; then
    echo "$(date '+%F %T') generating fixed-k 2-bin diagnostic config."
    "${PYTHON_BIN}" tools/generate_xr1_steam_configs.py \
        --data-root "${DATA_ROOT}" \
        --run-root "${RUN_ROOT}" \
        --config-dir "${CONFIG_PATH}" \
        --config-suffix _diag_b2 \
        --success-only \
        --num-bins 2 \
        --ensemble-size 1 \
        --value-max-steps "${BINARY_DIAGNOSTIC_STEPS}" \
        --value-save-interval "${BINARY_DIAGNOSTIC_STEPS}" \
        --micro-batch-size 8 \
        --global-batch-size 8 \
        --value-experiment-name steam_xr1_success_diag_b2 \
        --vision-model "${STEAM_VISION_MODEL}" \
        --language-model "${STEAM_LANGUAGE_MODEL}" \
        --pi05-model "${PI05_MODEL}" \
        --norm-stats-path "${ROBOCASA_NORM_STATS}"

    CRITIC_CONFIG=steam_value_model_sft_robocasa_xr1_diag_b2 \
    DIAGNOSTIC_STEPS="${BINARY_DIAGNOSTIC_STEPS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    REPO_PATH="${REPO_PATH}" RUN_ROOT="${RUN_ROOT}" \
        bash tools/run_success_critic_diagnostic.sh
fi

# ---------------------------------------------------------------------------
# Stage B: coarse temporal-distance diagnostic (8 bins, no length scaling).
# ---------------------------------------------------------------------------
if [[ "${RUN_8BIN_DIAGNOSTIC}" != "0" ]]; then
    NUM_BINS=8 \
    DIAGNOSTIC_STEPS="${EIGHT_BIN_DIAGNOSTIC_STEPS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    REPO_PATH="${REPO_PATH}" RUN_ROOT="${RUN_ROOT}" DATA_ROOT="${DATA_ROOT}" \
    STEAM_VISION_MODEL="${STEAM_VISION_MODEL}" \
    STEAM_LANGUAGE_MODEL="${STEAM_LANGUAGE_MODEL}" \
        bash tools/run_steam_multibin_diagnostic.sh
fi

unset STEAM_BINARY_STRICT_K || true

# ---------------------------------------------------------------------------
# Stage C: formal 32-bin normalized ensemble STEAM critic.
# ---------------------------------------------------------------------------
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
    --value-max-steps "${FORMAL_STEPS}" \
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
FORMAL_CHECKPOINT="${RUN_ROOT}/steam_value_training/steam_xr1_robocasa_value/checkpoints/global_step_${FORMAL_STEPS}/actor"

# The active YAML is the single source of truth for both mixture sampling and
# temporal-stride sampling.  PairDataset reads this exported seed via the runtime
# compatibility shim.
export STEAM_PAIR_SEED="$("${PYTHON_BIN}" - "${VALUE_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(int(cfg.data.get("seed", 42)))
PY
)"
echo "formal data.seed / STEAM_PAIR_SEED=${STEAM_PAIR_SEED}"

"${PYTHON_BIN}" tools/validate_steam_success_config.py --config "${VALUE_CONFIG}"
"${PYTHON_BIN}" tools/diagnose_steam_pairs.py \
    --config "${VALUE_CONFIG}" \
    --output-dir "${RUN_ROOT}/steam_diagnostics/formal_pretrain" \
    --samples-per-dataset 64 \
    --save-pairs 12

if [[ -n "${PIPELINE_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${PIPELINE_CUDA_VISIBLE_DEVICES}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

gpu_info="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
if ! printf '%s\n' "${gpu_info}" | awk -F, -v min="${MIN_FREE_MIB}" '$2+0 >= min {found=1} END {exit !found}'; then
    echo "ERROR: no visible GPU currently has at least ${MIN_FREE_MIB} MiB free" >&2
    printf '%s\n' "${gpu_info}" >&2
    exit 5
fi

echo "$(date '+%F %T') starting formal 32-bin STEAM value SFT."
"${PYTHON_BIN}" examples/offline_rl/advantage_labeling/steam/train_steam.py \
    --config-path "${CONFIG_PATH}" \
    --config-name steam_value_model_sft_robocasa_xr1

echo "$(date '+%F %T') formal value SFT finished; running fixed post-train gate."
if [[ ! -e "${FORMAL_CHECKPOINT}" ]]; then
    echo "ERROR: formal checkpoint not found: ${FORMAL_CHECKPOINT}" >&2
    exit 4
fi
"${PYTHON_BIN}" tools/evaluate_steam_multibin_checkpoint.py \
    --config "${VALUE_CONFIG}" \
    --checkpoint "${FORMAL_CHECKPOINT}" \
    --output "${RUN_ROOT}/steam_diagnostics/formal_posttrain_metrics.json" \
    --max-samples-per-dataset 256 \
    --device cuda \
    --seed "${STEAM_PAIR_SEED}" \
    --min-ce-improvement "${FORMAL_MIN_CE_IMPROVEMENT}" \
    --min-exact-improvement "${FORMAL_MIN_EXACT_IMPROVEMENT}" \
    --min-neighbor-improvement "${FORMAL_MIN_NEIGHBOR_IMPROVEMENT}"

# Only a critic that beats empirical/constant baselines may label advantages.
echo "$(date '+%F %T') starting STEAM advantage computation."
"${PYTHON_BIN}" examples/offline_rl/advantage_labeling/steam/process/compute_advantages_ensemble.py \
    --config-path "${CONFIG_PATH}" \
    --config-name steam_compute_advantages_robocasa_xr1

echo "$(date '+%F %T') advantage computation finished."

# ---------------------------------------------------------------------------
# Stage D: real XR-1 batch through Pi0.5 transforms + CFGRL flow forward.
# ---------------------------------------------------------------------------
echo "$(date '+%F %T') smoke-testing Pi0.5 CFG with a real XR-1 batch."
"${PYTHON_BIN}" tools/smoke_openpi_cfg_init.py \
    --config "${CFG_CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --device cuda

echo "$(date '+%F %T') starting Pi0.5 CFG-RL."
"${PYTHON_BIN}" examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py \
    --config-path "${CONFIG_PATH}" \
    --config-name cfg_rl_openpi_robocasa_xr1

echo "$(date '+%F %T') Pi0.5 CFG-RL finished."
