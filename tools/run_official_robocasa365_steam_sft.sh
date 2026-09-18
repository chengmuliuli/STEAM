#!/usr/bin/env bash
set -euo pipefail

# This wrapper only prepares the environment and selects the dataset.  The
# actual training entry point remains RLinf's official run_steam_sft.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLINF_ROOT="${RLINF_ROOT:-/data-cfs/data1/qiuguolin/RLinf_official_steam}"
PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-/data-cfs/data1/qiuguolin/tmp/steam_python_bin}"
VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES:-/data-cfs/data1/qiuguolin/RLinf/.venv/lib/python3.10/site-packages}"
CONFIG_NAME="steam_value_model_sft_robocasa365_pretrain"
CONFIG_SOURCE="${REPO_ROOT}/examples/offline_rl/config/${CONFIG_NAME}.yaml"
CONFIG_TARGET="${RLINF_ROOT}/examples/offline_rl/config/${CONFIG_NAME}.yaml"

export REPO_PATH="${RLINF_ROOT}"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/tools/official_lerobot_runtime:${REPO_ROOT}:${RLINF_ROOT}:${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data-cfs/data1/qiuguolin/cache/robocasa365_official_steam_task_ts}"
export STEAM_ROBOCASA_DATA="${STEAM_ROBOCASA_DATA:-/data-cfs/data1/qiuguolin/robocasa365-pretrain-mg-official-steam-task-ts}"
export STEAM_LOG_ROOT="${STEAM_LOG_ROOT:-/data-cfs/data1/qiuguolin/steam_official_robocasa365_pretrain}"

if [[ ! -f "${CONFIG_SOURCE}" ]]; then
  echo "Missing repository config: ${CONFIG_SOURCE}" >&2
  exit 2
fi
if [[ -e "${CONFIG_TARGET}" ]] && ! cmp -s "${CONFIG_SOURCE}" "${CONFIG_TARGET}"; then
  echo "Refusing to overwrite different config: ${CONFIG_TARGET}" >&2
  echo "Remove it or make it identical to the repository config first." >&2
  exit 2
fi
if [[ ! -e "${CONFIG_TARGET}" ]]; then
  mkdir -p "$(dirname "${CONFIG_TARGET}")"
  cp "${CONFIG_SOURCE}" "${CONFIG_TARGET}"
fi

echo "[STEAM] overlay_root=${REPO_ROOT}"
echo "[STEAM] rlinf_root=${RLINF_ROOT}"
echo "[STEAM] dataset=${STEAM_ROBOCASA_DATA}"
if command -v git >/dev/null 2>&1; then
  echo "[STEAM] overlay_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "[STEAM] rlinf_commit=$(git -C "${RLINF_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
python -c 'import sys; print("[STEAM] python=" + sys.executable); print("[STEAM] python_version=" + sys.version.split()[0])'
python -c 'import torch, torchvision, lerobot, datasets; print("[STEAM] torch=" + torch.__version__); print("[STEAM] torchvision=" + torchvision.__version__); print("[STEAM] lerobot=" + getattr(lerobot, "__version__", "unknown")); print("[STEAM] datasets=" + datasets.__version__)' 2>/dev/null || true

cd "${RLINF_ROOT}"
exec bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
  "${CONFIG_NAME}" "$@"

