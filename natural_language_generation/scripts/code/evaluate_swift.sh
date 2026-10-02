#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-output/python-swift64-Llama-2-7b}"
DATA_PATH="${DATA_PATH:-pissa-dataset}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
ADAPTER_PATH="${ADAPTER_PATH:-$OUTPUT_PATH/adapter_model}"
PREDICTIONS="$OUTPUT_PATH/python_response.jsonl"

if [[ ! -f "$ADAPTER_PATH/adapter_config.json" || ! -f "$ADAPTER_PATH/adapter_model.bin" ]]; then
  echo "SWIFT adapter not found in $ADAPTER_PATH; run scripts/code/train_swift.sh first." >&2
  exit 2
fi
if [[ ! -f "$DATA_PATH/python/test.json" ]]; then
  echo "Code evaluation data not found: $DATA_PATH/python/test.json" >&2
  exit 2
fi

rm -f "$PREDICTIONS" "$OUTPUT_PATH/humaneval.jsonl" "$OUTPUT_PATH/mbpp.jsonl"
python utils/gen_humaneval.py \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --data_path "$DATA_PATH" \
  --sub_task python \
  --output_file "$PREDICTIONS"
python utils/code_process.py --path "$PREDICTIONS"
evalplus.evaluate --dataset humaneval --samples "$OUTPUT_PATH/humaneval.jsonl"
evalplus.evaluate --dataset mbpp --samples "$OUTPUT_PATH/mbpp.jsonl"

