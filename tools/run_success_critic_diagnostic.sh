#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:?Set REPO_PATH to the RLinf checkout}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the converted rollout workspace}"
CRITIC_CONFIG="${CRITIC_CONFIG:-steam_value_model_sft_robocasa_xr1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
LOG_ROOT="${RUN_ROOT}/steam_value_training/queued_success_critic"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/queue.log") 2>&1

export PATH="${REPO_PATH}/.venv/bin:${PATH}"
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

CONFIG_PATH="${REPO_PATH}/examples/offline_rl/config/${CRITIC_CONFIG}.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: critic config not found: ${CONFIG_PATH}" >&2
    exit 2
fi

# This diagnostic is intentionally success-only. Fail fast if the generated
# config still contains recovery leaves, because that would re-introduce the
# temporal-label ambiguity we are trying to isolate.
if ! grep -q "normal_success" "${CONFIG_PATH}"; then
    echo "ERROR: ${CONFIG_PATH} does not contain normal_success datasets." >&2
    echo "Regenerate it with:" >&2
    echo "  python tools/generate_xr1_steam_configs.py --data-root \"${RUN_ROOT}\" --config-dir \"${REPO_PATH}/examples/offline_rl/config\" --success-only --num-bins 2" >&2
    exit 2
fi
if grep -q "recovery_" "${CONFIG_PATH}"; then
    echo "ERROR: ${CONFIG_PATH} still contains recovery datasets." >&2
    echo "Regenerate it with --success-only before running this diagnostic." >&2
    exit 2
fi

echo "$(date '+%F %T') queued success-only critic diagnostic"
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

cd "${REPO_PATH}"
echo "$(date '+%F %T') starting 2-bin overfit diagnostic with ${CRITIC_CONFIG}"

# Diagnostic goal: prove that the pair/label/model/loss plumbing is learnable.
# Keep the setup deliberately easy and remove long-run training confounders:
#   - 2 bins => random CE baseline ln(2) ~= 0.693
#   - single critic
#   - frozen SigLIP/Gemma backbones; train projectors + classification head
#   - no label smoothing and no LR warmup
#   - small single-GPU batch
bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
    "${CRITIC_CONFIG}" \
    actor.model.num_bins=2 \
    actor.model.ensemble_size=1 \
    actor.model.freeze_vision_encoder=true \
    actor.model.freeze_language_model=true \
    actor.model.label_smoothing=0.0 \
    actor.micro_batch_size=8 \
    actor.global_batch_size=8 \
    actor.optim.lr_warmup_steps=0 \
    runner.max_steps=512 \
    runner.save_interval=256 \
    actor.optim.total_training_steps=512

echo "$(date '+%F %T') success-only 2-bin critic diagnostic finished"
