# Environment provenance

The three environments intentionally follow the official repositories of the downstream frameworks instead of forcing one shared stack.

## Commonsense reasoning: DoRA

Upstream: [NVlabs/DoRA commonsense reasoning](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning).

The official instructions create Python 3.10 and install `commonsense_reasoning/requirements.txt`. This repository carries the same direct package set and versions. The upstream file installs Transformers from the `v4.36.0` Git tag; this release uses the equivalent `transformers==4.36.0` wheel so installation does not depend on Git. Upstream does not pin PyTorch or CUDA, so `environment.yml` adds the tested release lock PyTorch 2.1.0 with the CUDA 12.1 runtime. Hub, tokenizers, NumPy, protobuf, and tqdm are also pinned to prevent future resolver drift.

## Natural language generation: PiSSA

Upstream: [MuLabPKU/PiSSA](https://github.com/MuLabPKU/PiSSA).

The official quick start specifies Python 3.10, CUDA Toolkit 12.1, PyTorch 2.4.0, torchvision 0.19.0, `pytorch-cuda=12.1`, the repository requirements, and FlashAttention installed without build isolation. `environment.yml` keeps that stack and fixes versions for upstream entries that were unpinned. Run `natural_language_generation/scripts/install_upstream_extras.sh` after creating the environment to install `flash-attn==2.7.3` with the required no-build-isolation option.

The upstream dependency `vllm==0.6.2` remains installed for environment parity. SWIFT evaluation uses Transformers instead, because the released adapter has a custom tensor format that stock vLLM cannot load as an ordinary LoRA adapter.

## Image classification: GOAT

Upstream: [Facico/GOAT-PEFT](https://github.com/Facico/GOAT-PEFT).

The official instructions specify Python 3.9, the repository requirements (including PyTorch 2.4.0 and torchvision 0.19.0), then `flash-attn==2.7.3` and `human_eval`. `environment.yml` retains the official direct dependencies, pins previously unpinned versions, uses the CUDA 12.1 PyTorch build, and includes CUDA Toolkit 12.1 so FlashAttention can compile. Run `image_classification/scripts/install_upstream_extras.sh` after creating the environment.

`flash-attn` and `human_eval` are not imported by the CLIP image-classification entrypoint, but they are kept to match the documented GOAT environment. The training code leaves CLIP FlashAttention disabled, matching the upstream vision script.
