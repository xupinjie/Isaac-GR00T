#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Training script for Unitree G1 Dex1 robot on pick-and-place tasks.
# This script finetunes GR00T-N1.6-3B model on the G1_Dex1_3Picktask dataset.

set -x -e

# ============================================================================
# Configuration
# ============================================================================

# Environment variables
export WANDB_MODE=offline
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export CHECK_VERSION=0

rm -rf output_g1_dex1/checkpoint-*

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL_PATH="nvidia/GR00T-N1.6-3B"
DATASET_PATH="/home/scratch.pinjiex_hw/datas/G1_Dex1_3Picktask_Dataset_Merge_dedup"
MODALITY_CONFIG_PATH="${SCRIPT_DIR}/unitree_g1_dex1_config.py"
OUTPUT_DIR="${SCRIPT_DIR}/output_g1_dex1"

# Training parameters
NUM_GPUS=${NUM_GPUS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
SHARD_SIZE=${SHARD_SIZE:-512}

MAX_STEPS=${MAX_STEPS:-1200}
SAVE_STEPS=${SAVE_STEPS:-30000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
MASTER_PORT=${MASTER_PORT:-29501}

# Video decoding backend: "torchcodec", "decord", "ffmpeg", "opencv", "nvc" (NVIDIA GPU decoder)
VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}

# Multiprocessing context: "fork", "spawn", "forkserver"
# The async NVC path only demuxes GOPs in DataLoader workers; NVDEC runs in
# the training process, so fork is safe and avoids spawn serialization costs.
MULTIPROCESSING_CONTEXT=${MULTIPROCESSING_CONTEXT:-fork}

# ============================================================================
# Validate prerequisites
# ============================================================================

if [ ! -d "${DATASET_PATH}" ]; then
    echo "Error: Dataset path does not exist: ${DATASET_PATH}"
    exit 1
fi

if [ ! -f "${MODALITY_CONFIG_PATH}" ]; then
    echo "Error: Modality config file does not exist: ${MODALITY_CONFIG_PATH}"
    exit 1
fi

# Create output directory if not exists
mkdir -p "${OUTPUT_DIR}"

# Log file path with timestamp and training configuration
# Format: training_YYYYMMDD_HHMMSS_gpu{N}_bs{B}_shard{S}_{backend}.log
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S)_gpu${NUM_GPUS}_bs${GLOBAL_BATCH_SIZE}_shard${SHARD_SIZE}_${VIDEO_BACKEND}.log"

# ============================================================================
# Launch training
# ============================================================================

echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Base Model: ${BASE_MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "Max Steps: ${MAX_STEPS}"
echo "Video Backend: ${VIDEO_BACKEND}"
echo "Multiprocessing Context: ${MULTIPROCESSING_CONTEXT}"
echo "Log File: ${LOG_FILE}"
echo "=========================================="

# Always use torchrun to initialize distributed environment (required by factory.py barrier)
# Use uv run to ensure .venv environment is used
# Use tee to save logs to file while displaying on console
echo "Starting training with ${NUM_GPUS} GPU(s)..."
echo "Logs will be saved to: ${LOG_FILE}"
PYTHONPATH=. uv run torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG_PATH}" \
    --num-gpus "${NUM_GPUS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
    --shard-size "${SHARD_SIZE}" \
    --video-backend "${VIDEO_BACKEND}" \
    --multiprocessing-context "${MULTIPROCESSING_CONTEXT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "Training completed! Log saved to: ${LOG_FILE}"
