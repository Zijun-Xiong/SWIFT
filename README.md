# SWIFT: Breaking Information Islands in Sparse Tuning via Small-World Connectivity

Official implementation of **Small-World Induced Fine-Tuning (SWIFT)**. The repository reproduces the three downstream experiment groups in the paper.

| Experiment | Backbone | Training data | Evaluation |
| --- | --- | --- | --- |
| Commonsense reasoning | LLaMA3-8B | Commonsense-170k | BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA |
| Natural language generation | LLaMA2-7B | 100K MetaMathQA / 100K CodeFeedback | GSM8K, MATH, HumanEval, MBPP |
| Image classification | CLIP ViT-B/32 | Per-dataset training split | Cars, DTD, GTSRB, RESISC45, SVHN |

## SWIFT parameterization

Each adapted linear layer keeps its pretrained weight frozen and learns `N` block matrices. The local block outputs are routed by a fixed uniform-scattering permutation:

```python
indices.reshape(N, block_width).transpose(0, 1).reshape(-1)
```

The permutation is deterministic, is applied to the **output of the block-diagonal update only**, and is stored as a persistent buffer. No random input permutation or inverse permutation is used. Dimensions that are not divisible by `N` are padded internally and sliced back after routing. The paper setting is `N=64`.

## Repository layout

```text
commonsense_reasoning/       LLaMA3-8B + Commonsense-170k
natural_language_generation/ LLaMA2-7B + MetaMathQA/CodeFeedback
image_classification/        CLIP ViT-B/32 + five vision datasets
tests/                       structural and permutation checks
```

Each experiment folder contains its own `README.md`, dependencies, and launch scripts. Model weights, datasets, caches, and generated outputs are intentionally excluded. Use a separate virtual environment for each experiment because their framework baselines intentionally pin different Transformers/Accelerate versions.

## Supported platform and reproducible environments

The supported training platform is Linux x86-64 with NVIDIA GPUs and a driver capable of running CUDA 12.1 builds. Native Windows is not supported by the Bash and DeepSpeed launchers. The PyTorch packages include the CUDA runtime; a system CUDA toolkit is only needed if a local package attempts to compile optional CUDA extensions.

Do **not** install all three experiments into one environment. Their reference frameworks require different versions:

| Experiment | Conda environment | Python | PyTorch | CUDA runtime | Other critical pins |
| --- | --- | --- | --- | --- | --- |
| Commonsense | `swift-commonsense` | 3.10 | 2.1.0 | 12.1 | Transformers 4.36.0, Accelerate 0.25.0 |
| NLG | `swift-nlg` | 3.10 | 2.4.0 + torchvision 0.19.0 | 12.1 | Transformers 4.45.1, Accelerate 1.2.1, DeepSpeed 0.15.4 |
| Vision | `swift-vision` | 3.9 | 2.4.0 + torchvision 0.19.0 | 12.1 | Transformers 4.44.0, PEFT 0.14.0, DeepSpeed 0.15.4 |

Create exactly one environment for the experiment you want to run:

```bash
conda env create -f commonsense_reasoning/environment.yml
# or: conda env create -f natural_language_generation/environment.yml
# or: conda env create -f image_classification/environment.yml
```

Activate it and run the fail-fast diagnostic, for example:

```bash
conda activate swift-commonsense
python scripts/doctor.py --experiment commonsense_reasoning
```

Use `swift-nlg` with `natural_language_generation`, or `swift-vision` with `image_classification`. The diagnostic checks Python, every critical package version, the CUDA build, visible GPUs, launcher commands, and whether the repository-local SWIFT implementation imports with `N=64`. The individual `requirements.txt` files are for an already-created environment with the correct PyTorch build; `environment.yml` is the recommended fresh-install path.

For NLG and vision, install the upstream FlashAttention extra before running the diagnostic:

```bash
bash natural_language_generation/scripts/install_upstream_extras.sh
# or: bash image_classification/scripts/install_upstream_extras.sh
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the official DoRA/PiSSA/GOAT source instructions and the few additional pins used to make dependency resolution repeatable.

The LLaMA2 and LLaMA3 repositories are gated. Accept Meta's model terms and run `huggingface-cli login` before training. Model and dataset paths may also point to already-downloaded local copies.

## Quick verification

From the repository root:

```bash
python tests/test_swift_structure.py

# in swift-commonsense
python tests/smoke_language_peft.py --experiment commonsense_reasoning
python tests/smoke_commonsense_checkpoint.py

# in swift-nlg
python tests/smoke_language_peft.py --experiment natural_language_generation
python tests/smoke_nlg_checkpoint.py

# in swift-vision
python tests/smoke_vision_peft.py
```

Then follow the README in the desired experiment folder. The launch scripts expose model, data, output, GPU, and block-count settings through environment variables.

These checks cover source parsing, the exact reshape-transpose-reshape routing, non-divisible dimensions, gradients, adapter save/reload, checkpoint resume, and launcher argument parsing without downloading large models. A full 7B/8B training run still requires the gated weights, complete datasets, and the GPU resources listed in the experiment README; exact paper accuracies can also vary with hardware and nondeterministic kernels.

## Acknowledgements and licenses

The language experiments build on PEFT-based training code and the vision experiments build on the GOAT framework. See `THIRD_PARTY_NOTICES.md` and the license files retained inside each subproject. Repository-level code is released under Apache-2.0 unless a subdirectory states otherwise.

