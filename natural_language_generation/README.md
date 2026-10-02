# Natural language generation

This experiment fine-tunes LLaMA2-7B on 100K examples from MetaMathQA and CodeFeedback. It evaluates mathematical reasoning on GSM8K and MATH and code generation on HumanEval and MBPP.

The SWIFT layer is implemented in `peft/src/peft/tuners/lora.py`. `train.py` constructs `LoraConfig(N=swift_num_blocks, ...)`; the paper setting is `N=64`.

## Setup and data

```bash
cd SWIFT-GitHub
conda env create -f natural_language_generation/environment.yml
conda activate swift-nlg
bash natural_language_generation/scripts/install_upstream_extras.sh
python scripts/doctor.py --experiment natural_language_generation
python tests/test_swift_structure.py
python tests/smoke_language_peft.py --experiment natural_language_generation
python tests/smoke_nlg_checkpoint.py

cd natural_language_generation
huggingface-cli login
huggingface-cli download --repo-type dataset fxmeng/pissa-dataset --local-dir pissa-dataset
```

This follows the official PiSSA setup: Python 3.10, CUDA Toolkit 12.1, PyTorch 2.4.0, torchvision 0.19.0, Transformers 4.45.1, Accelerate 1.2.1, DeepSpeed 0.15.4, vLLM 0.6.2, and FlashAttention. The dataset is available from [fxmeng/pissa-dataset](https://huggingface.co/datasets/fxmeng/pissa-dataset); the download must contain the `metamath/` and `python/` configurations. Accept access to `meta-llama/Llama-2-7b-hf` before logging in. `BASE_MODEL` may instead be a local model directory.

## Train

```bash
bash scripts/math/train_swift.sh
bash scripts/code/train_swift.sh
```

Both scripts default to two GPUs (`GPU_IDS=0,1`), LLaMA2-7B, `N=64`, 100K training examples, one epoch, and learning rate `2e-4`. Override `BASE_MODEL`, `DATA_PATH`, `OUTPUT_PATH`, `GPU_IDS`, or `SWIFT_NUM_BLOCKS` as needed.

Run these commands from `natural_language_generation/`. A concrete local-path example is:

```bash
BASE_MODEL=/models/Llama-2-7b-hf \
DATA_PATH=/data/pissa-dataset \
OUTPUT_PATH=/runs/swift-math \
GPU_IDS=0,1 bash scripts/math/train_swift.sh
```

## Evaluate

Training writes the final SWIFT adapter to `OUTPUT_PATH/adapter_model`; the base-model weights are not duplicated. Generate predictions with the same base model plus that adapter, then run the included evaluators with:

```bash
bash scripts/math/evaluate_swift.sh
bash scripts/code/evaluate_swift.sh
```

The evaluation scripts use `BASE_MODEL`, `OUTPUT_PATH`, `ADAPTER_PATH`, and `DATA_PATH`. Keep `BASE_MODEL` identical to training. They intentionally load the repository's local PEFT implementation because standard PEFT does not know the custom SWIFT tensor format.

```bash
BASE_MODEL=/models/Llama-2-7b-hf \
DATA_PATH=/data/pissa-dataset \
OUTPUT_PATH=/runs/swift-math \
ADAPTER_PATH=/runs/swift-math/adapter_model \
bash scripts/math/evaluate_swift.sh
```

`evaluate_swift.sh` for code runs EvalPlus against model-generated Python. Run that evaluator inside an isolated container or other sandbox; do not execute untrusted generations on a sensitive host.

