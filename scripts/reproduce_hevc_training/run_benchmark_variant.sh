#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 VARIANT VIDEO_BACKEND MASTER_PORT" >&2
    exit 2
fi

variant=$1
video_backend=$2
master_port=$3

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=${PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}
dataset_path=${DATASET_PATH:?Set DATASET_PATH to the converted HEVC dataset}
benchmark_root=${BENCHMARK_ROOT:?Set BENCHMARK_ROOT to the experiment output directory}
base_model_path=${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}
timing_dir=${benchmark_root}/raw/${variant}
output_dir=${benchmark_root}/model_output/${variant}
log_path=${benchmark_root}/logs/${variant}.log

for path in "${timing_dir}" "${output_dir}" "${log_path}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to mix results with existing path: ${path}" >&2
        exit 1
    fi
done
mkdir -p "${timing_dir}" "${output_dir}" "$(dirname "${log_path}")"

export PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/venvs/isaac-groot-a100}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/isaac-groot-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch-extensions}
export LD_LIBRARY_PATH=${project_root}/.runtime/nvidia-driver:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export WANDB_MODE=offline
export NO_ALBUMENTATIONS_UPDATE=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export CHECK_VERSION=0
export PROFILE_VARIANT=${variant}
export PROFILE_TIMING_DIR=${timing_dir}

start_epoch=$(date +%s)
cd "${project_root}"
PYTHONPATH="${project_root}" uv run --no-sync torchrun \
    --nproc_per_node=8 \
    --master_port="${master_port}" \
    "${project_root}/scripts/profile_step_timing.py" \
    --base-model-path "${base_model_path}" \
    --dataset-path "${dataset_path}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${project_root}/unitree_g1_dex1_config.py" \
    --num-gpus 8 \
    --global-batch-size 256 \
    --max-steps 200 \
    --save-steps 30000 \
    --save-total-limit 1 \
    --warmup-ratio 0.05 \
    --dataloader-num-workers 4 \
    --shard-size 512 \
    --video-backend "${video_backend}" \
    --multiprocessing-context fork \
    --output-dir "${output_dir}" \
    2>&1 | tr '\r' '\n' | sed -u \
        -e '/Recovered pix_fmt from SPS extradata/d' \
        -e '/enabling stream aware allocations!/d' \
        -e '/^$/d' | tee "${log_path}"
end_epoch=$(date +%s)
printf '%s\n' "$((end_epoch - start_epoch))" >"${benchmark_root}/${variant}_process_wall_seconds.txt"
