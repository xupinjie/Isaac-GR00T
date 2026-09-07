#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=${PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}
source_dataset=${SOURCE_DATASET:-${project_root}/datasets/G1_Dex1_MountCameraRedGripper_Dataset}
hevc_dataset=${HEVC_DATASET:-${project_root}/datasets/G1_Dex1_MountCameraRedGripper_Dataset_HEVC}
benchmark_root=${BENCHMARK_ROOT:-${project_root}/output_hevc_reproduction}
base_model_path=${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the downloaded GR00T-N1.6-3B model}

export DEBIAN_FRONTEND=noninteractive
if ! command -v ffmpeg >/dev/null || ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx265; then
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg
fi

cd "${project_root}"
python scripts/reproduce_hevc_training/prepare_hevc_dataset.py \
    "${source_dataset}" "${hevc_dataset}" \
    --gop-size 2 --preset medium --crf 23 --jobs 32 --x265-pools 4
install -D -m 0644 \
    scripts/reproduce_hevc_training/g1_dex1_modality.json \
    "${hevc_dataset}/meta/modality.json"

export PATH=/root/.local/bin:${PATH}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/venvs/isaac-groot-a100}
if ! command -v uv >/dev/null; then
    python -m pip install --user 'uv>=0.8.4'
fi
uv sync --frozen --inexact --python 3.10

export PROJECT_ROOT=${project_root}
export DATASET_PATH=${hevc_dataset}
export BENCHMARK_ROOT=${benchmark_root}
export BASE_MODEL_PATH=${base_model_path}

bash scripts/reproduce_hevc_training/install_accv_decoder.sh
bash scripts/reproduce_hevc_training/run_benchmark_variant.sh cpu_torchcodec torchcodec 29811
bash scripts/reproduce_hevc_training/run_benchmark_variant.sh async_gop nvc 29812
python scripts/aggregate_step_timing.py "${benchmark_root}" \
    --variants cpu_torchcodec async_gop
python scripts/reproduce_hevc_training/summarize_benchmark.py "${benchmark_root}" \
    | tee "${benchmark_root}/comparison_table.md"
rm -rf "${benchmark_root}/model_output"
