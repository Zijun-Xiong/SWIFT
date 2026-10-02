#!/usr/bin/env python3
"""Fail-fast environment checks for one SWIFT experiment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ENVIRONMENTS = {
    "commonsense_reasoning": {
        "python": (3, 10),
        "torch": "2.1.0",
        "cuda": "12.1",
        "packages": {
            "accelerate": "0.25.0",
            "appdirs": "1.4.4",
            "bitsandbytes": "0.41.3.post2",
            "black": "23.12.0",
            "datasets": "2.15.0",
            "fire": "0.5.0",
            "gradio": "4.9.0",
            "huggingface-hub": "0.19.4",
            "numpy": "1.26.4",
            "protobuf": "4.25.5",
            "scipy": "1.11.4",
            "sentencepiece": "0.1.99",
            "tokenizers": "0.15.0",
            "tqdm": "4.66.4",
            "transformers": "4.36.0",
        },
        "imports": ("accelerate", "datasets", "fire", "transformers"),
        "commands": (),
        "recommended_gpus": 1,
        "bf16": False,
    },
    "natural_language_generation": {
        "python": (3, 10),
        "torch": "2.4.0",
        "cuda": "12.1",
        "packages": {
            "accelerate": "1.2.1",
            "attrdict": "2.0.1",
            "datasets": "3.2.0",
            "deepspeed": "0.15.4",
            "evalplus": "0.3.1",
            "flash-attn": "2.7.3",
            "fraction": "2.2.0",
            "human_eval": "1.0.3",
            "huggingface-hub": "0.25.2",
            "ninja": "1.11.1.1",
            "numpy": "1.26.4",
            "peft": "0.14.0",
            "protobuf": "4.25.5",
            "sentencepiece": "0.2.0",
            "tensorboardX": "2.6.2.2",
            "tokenizers": "0.20.3",
            "torchvision": "0.19.0",
            "tqdm": "4.66.4",
            "transformers": "4.45.1",
            "vllm": "0.6.2",
        },
        "imports": ("accelerate", "datasets", "deepspeed", "transformers"),
        "commands": ("deepspeed", "torchrun"),
        "recommended_gpus": 2,
        "bf16": True,
    },
    "image_classification": {
        "python": (3, 9),
        "torch": "2.4.0",
        "cuda": "12.1",
        "packages": {
            "accelerate": "0.33.0",
            "datasets": "2.19.2",
            "deepspeed": "0.15.4",
            "fire": "0.6.0",
            "flash-attn": "2.7.3",
            "h5py": "3.12.1",
            "huggingface-hub": "0.25.2",
            "human_eval": "1.0.3",
            "hydra-core": "1.3.2",
            "lightning": "2.2.5",
            "matplotlib": "3.9.2",
            "multipledispatch": "1.0.0",
            "ninja": "1.11.1.1",
            "numpy": "1.26.4",
            "omegaconf": "2.3.0",
            "pandas": "2.2.3",
            "peft": "0.14.0",
            "Pillow": "10.4.0",
            "rich": "13.9.4",
            "scikit-learn": "1.5.2",
            "scipy": "1.13.1",
            "sentencepiece": "0.2.0",
            "setuptools": "69.5.1",
            "tokenizers": "0.19.1",
            "torchmetrics": "1.4.0.post0",
            "torchvision": "0.19.0",
            "tqdm": "4.66.4",
            "transformers": "4.44.0",
            "wandb": "0.18.3",
        },
        "imports": ("accelerate", "deepspeed", "pandas", "peft", "PIL", "scipy", "torchvision", "transformers"),
        "commands": ("torchrun",),
        "recommended_gpus": 4,
        "bf16": True,
    },
}


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[ OK ] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def normalized(version: str) -> str:
    return version.split("+", 1)[0]


def check_version(reporter: Reporter, distribution: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        reporter.fail(f"missing package: {distribution}=={expected}")
        return
    if normalized(actual) == expected:
        reporter.ok(f"{distribution}=={actual}")
    else:
        reporter.fail(f"{distribution}: expected {expected}, found {actual}")


def check_local_swift(reporter: Reporter, experiment: str) -> None:
    try:
        if experiment in {"commonsense_reasoning", "natural_language_generation"}:
            local_src = ROOT / experiment / "peft" / "src"
            sys.path.insert(0, str(local_src))
            peft = importlib.import_module("peft")
            loaded_from = Path(peft.__file__).resolve()
            if local_src.resolve() not in loaded_from.parents:
                raise RuntimeError(f"loaded non-local PEFT from {loaded_from}")
            config = peft.LoraConfig(N=64, target_modules=["q_proj"], task_type="CAUSAL_LM")
            if config.N != 64:
                raise RuntimeError("LoraConfig did not preserve N=64")
        else:
            goat_root = ROOT / experiment / "goat"
            sys.path.insert(0, str(goat_root))
            peta = importlib.import_module("peta")
            peta.src.register_swift_with_peft()
            config = peta.src.GOATConfig(swift_num_blocks=64, target_modules=["q_proj"])
            if config.swift_num_blocks != 64:
                raise RuntimeError("GOATConfig did not preserve N=64")
        reporter.ok("repository-local SWIFT implementation imports and accepts N=64")
    except Exception as exc:  # show the actionable import/configuration error
        reporter.fail(f"local SWIFT import failed: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(ENVIRONMENTS))
    parser.add_argument("--allow-cpu", action="store_true", help="Permit a CPU-only smoke-test host.")
    parser.add_argument(
        "--allow-unsupported-os",
        action="store_true",
        help="Permit import-only checks outside the supported Linux training platform.",
    )
    args = parser.parse_args()

    expected = ENVIRONMENTS[args.experiment]
    reporter = Reporter()

    actual_python = sys.version_info[:2]
    if actual_python == expected["python"]:
        reporter.ok(f"Python {platform.python_version()}")
    else:
        reporter.fail(
            f"Python {expected['python'][0]}.{expected['python'][1]} is required; "
            f"found {platform.python_version()}"
        )

    if platform.system() == "Linux":
        reporter.ok("Linux training platform")
    elif args.allow_unsupported_os:
        reporter.warn(f"{platform.system()} is suitable only for local smoke tests; use Linux for training")
    else:
        reporter.fail(f"{platform.system()} is unsupported for the Bash/DeepSpeed training workflow; use Linux")

    check_version(reporter, "torch", expected["torch"])
    for distribution, version in expected["packages"].items():
        check_version(reporter, distribution, version)

    for module in expected["imports"]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            reporter.fail(f"cannot import {module}: {type(exc).__name__}: {exc}")
        else:
            reporter.ok(f"import {module}")

    for command in expected["commands"]:
        executable = shutil.which(command)
        if executable:
            reporter.ok(f"command {command}: {executable}")
        else:
            reporter.fail(f"command not found on PATH: {command}")

    try:
        import torch

        built_cuda = torch.version.cuda
        if built_cuda and built_cuda.startswith(expected["cuda"]):
            reporter.ok(f"PyTorch CUDA runtime {built_cuda}")
        else:
            reporter.fail(f"PyTorch must be built for CUDA {expected['cuda']}; found {built_cuda or 'CPU-only'}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(gpu_count))
            reporter.ok(f"CUDA visible: {gpu_count} GPU(s): {names}")
            if gpu_count < expected["recommended_gpus"]:
                reporter.warn(
                    f"paper launcher defaults to {expected['recommended_gpus']} GPU(s); "
                    "override GPU_IDS and batch settings for this host"
                )
            if expected["bf16"] and not torch.cuda.is_bf16_supported():
                reporter.fail("this experiment enables bf16, but the visible GPU does not support it")
        elif args.allow_cpu:
            reporter.warn("CUDA is unavailable; only CPU smoke tests can run")
        else:
            reporter.fail("CUDA is unavailable; check the NVIDIA driver and CUDA-visible devices")
    except Exception as exc:
        reporter.fail(f"PyTorch runtime check failed: {type(exc).__name__}: {exc}")

    check_local_swift(reporter, args.experiment)

    print(f"\nSummary: {reporter.failures} failure(s), {reporter.warnings} warning(s).")
    if reporter.failures:
        print("Environment is NOT ready. Recreate it from the experiment's environment.yml.")
        return 1
    print("Environment is ready for the repository smoke tests and experiment launcher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
