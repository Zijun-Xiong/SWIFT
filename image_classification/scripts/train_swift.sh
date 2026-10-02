#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/goat"

MODEL="${MODEL:-openai/clip-vit-base-patch32}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/results}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-19987}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SWIFT_NUM_BLOCKS="${SWIFT_NUM_BLOCKS:-64}"
REPORT_TO="${REPORT_TO:-none}"
TASKS="${TASKS:-Cars DTD GTSRB RESISC45 SVHN}"

# Download the two redistributable datasets and fail early on incomplete manual layouts.
python download_data.py --data_root "$DATA_ROOT" --tasks $TASKS

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
GRAD_ACC=$(( TOTAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NUM_GPUS ))
if (( TOTAL_BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * NUM_GPUS) != 0 )); then
  echo "TOTAL_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE * NUM_GPUS" >&2
  exit 2
fi
if (( GRAD_ACC < 1 )); then
  echo "TOTAL_BATCH_SIZE must be at least PER_DEVICE_BATCH_SIZE * NUM_GPUS" >&2
  exit 2
fi

declare -A EPOCHS=(
  [Cars]=35
  [DTD]=76
  [GTSRB]=11
  [RESISC45]=15
  [SVHN]=4
)

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$RESULT_ROOT"
for task in $TASKS; do
  if [[ -z "${EPOCHS[$task]+x}" ]]; then
    echo "Unknown task: $task" >&2
    exit 2
  fi

  run_name="clip-vit-b32-${task}-swift-N${SWIFT_NUM_BLOCKS}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
    --nnodes=1 \
    --nproc-per-node="$NUM_GPUS" \
    --master-port="$MASTER_PORT" \
    train_vit.py \
    --lora src.goat \
    --task "$task" \
    --bz "$PER_DEVICE_BATCH_SIZE" \
    --model "$MODEL" \
    --gacc "$GRAD_ACC" \
    --ep "${EPOCHS[$task]}" \
    --lr "$LEARNING_RATE" \
    --prj "$run_name" \
    --swift_num_blocks "$SWIFT_NUM_BLOCKS" \
    --data_root "$DATA_ROOT" \
    --report_to "$REPORT_TO" \
    --result "$RESULT_ROOT" \
    --output "$OUTPUT_ROOT/$run_name" \
    --seed 0
done

