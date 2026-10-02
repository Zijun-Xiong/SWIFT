#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-output/math-swift64-Llama-2-7b}"
DATA_PATH="${DATA_PATH:-pissa-dataset}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
ADAPTER_PATH="${ADAPTER_PATH:-$OUTPUT_PATH/adapter_model}"
PREDICTIONS="$OUTPUT_PATH/metamath_response.jsonl"

if [[ ! -f "$ADAPTER_PATH/adapter_config.json" || ! -f "$ADAPTER_PATH/adapter_model.bin" ]]; then
  echo "SWIFT adapter not found in $ADAPTER_PATH; run scripts/math/train_swift.sh first." >&2
  exit 2
fi
if [[ ! -f "$DATA_PATH/metamath/test.json" ]]; then
  echo "MetaMath evaluation data not found: $DATA_PATH/metamath/test.json" >&2
  exit 2
fi

rm -f "$PREDICTIONS"
python utils/gen_math.py \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --data_path "$DATA_PATH" \
  --sub_task metamath \
  --output_file "$PREDICTIONS"
python utils/test_acc.py --input_file "$PREDICTIONS"

