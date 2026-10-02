# Commonsense reasoning

This experiment fine-tunes LLaMA3-8B on Commonsense-170k and evaluates eight benchmarks: BoolQ, PIQA, SocialIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA.

The SWIFT layer is implemented in `peft/src/peft/tuners/lora.py`. `finetune.py` constructs `LoraConfig(N=swift_num_blocks, ...)`; the paper setting is `N=64`.

## Setup

```bash
cd SWIFT-GitHub
conda env create -f commonsense_reasoning/environment.yml
conda activate swift-commonsense
python scripts/doctor.py --experiment commonsense_reasoning
python tests/test_swift_structure.py
python tests/smoke_language_peft.py --experiment commonsense_reasoning
python tests/smoke_commonsense_checkpoint.py
```

This pins Python 3.10, PyTorch 2.1.0 with CUDA 12.1, Transformers 4.36.0, Accelerate 0.25.0, and the compatible Hub/tokenizers releases. The default scripts use 16-bit weights. Install a CUDA-compatible `bitsandbytes` separately only if you enable `--load_8bit`.

Accept access to `meta-llama/Meta-Llama-3-8B`, then authenticate once with `huggingface-cli login`. `BASE_MODEL` may instead be an existing local model directory.

## Data

Download [`commonsense_170k.json`](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/ft-training_set) into `ft-training_set/`, and place the [eight evaluation JSON files](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) in `dataset/`.

The evaluator requires this exact layout:

```text
ft-training_set/commonsense_170k.json
dataset/boolq/test.json
dataset/piqa/test.json
dataset/social_i_qa/test.json
dataset/hellaswag/test.json
dataset/winogrande/test.json
dataset/ARC-Easy/test.json
dataset/ARC-Challenge/test.json
dataset/openbookqa/test.json
```

## Train and evaluate

```bash
cd commonsense_reasoning
bash scripts/train_swift.sh
bash scripts/evaluate_swift.sh
```

Defaults reproduce the paper configuration: LLaMA3-8B, `N=64`, two epochs, learning rate `5e-5`, and target modules `q_proj`, `k_proj`, `v_proj`, `up_proj`, and `down_proj`. Paths and GPU selection can be overridden with the environment variables at the top of each script.

The default run uses one 24 GB-class GPU. Example with local paths:

```bash
BASE_MODEL=/models/Meta-Llama-3-8B \
DATA_PATH=/data/commonsense_170k.json \
OUTPUT_DIR=/runs/swift-commonsense \
CUDA_VISIBLE_DEVICES=0 bash scripts/train_swift.sh

BASE_MODEL=/models/Meta-Llama-3-8B \
ADAPTER_PATH=/runs/swift-commonsense \
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_swift.sh
```

