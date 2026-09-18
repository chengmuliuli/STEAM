#!/usr/bin/env bash
set -euo pipefail

# This wrapper only prepares the environment and selects the dataset.  The
# actual training entry point remains RLinf's official run_steam_sft.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="${RLINF_ROOT:-/data-cfs/data1/qiuguolin/RLinf_official_steam}"
PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-/data-cfs/data1/qiuguolin/tmp/steam_python_bin}"
VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES:-/data-cfs/data1/qiuguolin/RLinf/.venv/lib/python3.10/site-packages}"

export REPO_PATH="${RLINF_ROOT}"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/tools/official_lerobot_runtime:${REPO_ROOT}:${RLINF_ROOT}:${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data-cfs/data1/qiuguolin/cache/robocasa365_official_steam_task_ts}"
export STEAM_ROBOCASA_DATA="${STEAM_ROBOCASA_DATA:-/data-cfs/data1/qiuguolin/robocasa365-pretrain-mg-official-steam-task-ts}"
export STEAM_LOG_ROOT="${STEAM_LOG_ROOT:-/data-cfs/data1/qiuguolin/steam_official_robocasa365_pretrain}"

cd "${RLINF_ROOT}"
exec bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
  steam_value_model_sft_robocasa365_pretrain "$@"

