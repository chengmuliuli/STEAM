#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:-/path/to/RLinf}"
CONFIG_PATH="${REPO_PATH}/examples/offline_rl/config"
RUN_ROOT="${RUN_ROOT:-/path/to/steam_xr1_v30_copy}"
LOG_ROOT="${RUN_ROOT}/pipeline"
mkdir -p "${LOG_ROOT}"

exec > >(tee -a "${LOG_ROOT}/pipeline.log") 2>&1

export PYTHONPATH="${REPO_PATH}/tools:${REPO_PATH}"
export PYTHONPATH="${REPO_PATH}/.venv/lib/python3.10/site-packages:${REPO_PATH}/tools:${REPO_PATH}"
export REPO_PATH
export HF_HOME="${RUN_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${RUN_ROOT}/hf_datasets_cache"
export TRANSFORMERS_CACHE="${RUN_ROOT}/hf_cache/transformers"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off
export RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:47999}"
export RAY_TMPDIR="/tmp/qgl-ray-steam"
mkdir -p "${RAY_TMPDIR}"

wait_for_gpu0() {
    while true; do
        free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n 1 || true)"
        if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= 16000 )); then
            echo "$(date '+%F %T') GPU0 has ${free_mib} MiB free."
            return 0
        fi
        echo "$(date '+%F %T') waiting for GPU0: free=${free_mib:-unknown} MiB; need 16000 MiB."
        sleep 60
    done
}

cd "${REPO_PATH}"
echo "$(date '+%F %T') XR-1 STEAM pipeline queued."

wait_for_gpu0
export CUDA_VISIBLE_DEVICES=0
echo "$(date '+%F %T') starting STEAM value SFT."
/usr/bin/python3.10 examples/offline_rl/advantage_labeling/steam/train_steam.py --config-path "${CONFIG_PATH}" --config-name steam_value_model_sft_robocasa_xr1
echo "$(date '+%F %T') value SFT finished."

wait_for_gpu0
echo "$(date '+%F %T') starting STEAM advantage computation."
/usr/bin/python3.10 examples/offline_rl/advantage_labeling/steam/process/compute_advantages_ensemble.py --config-path "${CONFIG_PATH}" --config-name steam_compute_advantages_robocasa_xr1
echo "$(date '+%F %T') advantage computation finished."

wait_for_gpu0
echo "$(date '+%F %T') starting Pi05 CFG-RL."
/usr/bin/python3.10 examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py --config-path "${CONFIG_PATH}" --config-name cfg_rl_openpi_robocasa_xr1
echo "$(date '+%F %T') Pi05 CFG-RL finished."
