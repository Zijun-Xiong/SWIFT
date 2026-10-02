#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
OUTPUT_PATH="${OUTPUT_PATH:-output/python-swift64-Llama-2-7b}"
DATA_PATH="${DATA_PATH:-pissa-dataset}"
GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-16971}"
SWIFT_NUM_BLOCKS="${SWIFT_NUM_BLOCKS:-64}"

if [[ ! -f "$DATA_PATH/python/train.json" ]]; then
  echo "Code training data not found: $DATA_PATH/python/train.json" >&2
  echo "Download fxmeng/pissa-dataset as described in natural_language_generation/README.md." >&2
  exit 2
fi

deepspeed --master_port="$MASTER_PORT" --include="localhost:$GPU_IDS" train.py \
  --deepspeed configs/ds_config_zero2_no_offload.json \
  --model_name_or_path "$BASE_MODEL" \
  --full_finetune False \
  --bf16 \
  --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --swift_num_blocks "$SWIFT_NUM_BLOCKS" \
  --data_path "$DATA_PATH" \
  --sub_task python:100000 \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir "$OUTPUT_PATH" \
  --num_train_epochs 1 \
  --model_max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --lr_scheduler_type cosine \
  --report_to tensorboard

