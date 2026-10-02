#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-./trained_models/llama3_swift_N64}"
MODEL_ALIAS="${MODEL_ALIAS:-LLaMA3-8B}"
if [[ ! -f "$ADAPTER_PATH/adapter_config.json" || ! -f "$ADAPTER_PATH/adapter_model.bin" ]]; then
  echo "SWIFT adapter not found in $ADAPTER_PATH; run scripts/train_swift.sh first." >&2
  exit 2
fi

for dataset in openbookqa ARC-Easy social_i_qa ARC-Challenge winogrande piqa boolq hellaswag; do
  if [[ ! -f "dataset/$dataset/test.json" ]]; then
    echo "Evaluation data not found: dataset/$dataset/test.json" >&2
    echo "See commonsense_reasoning/README.md for the required layout." >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python commonsense_evaluate.py \
    --model "$MODEL_ALIAS" \
    --adapter LoRA \
    --dataset "$dataset" \
    --batch_size 1 \
    --base_model "$BASE_MODEL" \
    --lora_weights "$ADAPTER_PATH" | tee "$ADAPTER_PATH/${dataset}.txt"
done

