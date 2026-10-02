#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
DATA_PATH="${DATA_PATH:-./ft-training_set/commonsense_170k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./trained_models/llama3_swift_N64}"
SWIFT_NUM_BLOCKS="${SWIFT_NUM_BLOCKS:-64}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Training data not found: $DATA_PATH" >&2
  echo "See commonsense_reasoning/README.md for the required download." >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python finetune.py \
  --base_model "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 16 \
  --micro_batch_size 2 \
  --num_epochs 2 \
  --learning_rate 5e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --eval_step 80 \
  --save_step 80 \
  --adapter_name lora \
  --swift_num_blocks "$SWIFT_NUM_BLOCKS" \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --use_gradient_checkpointing \
  --train_on_inputs False

