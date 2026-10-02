#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "FlashAttention installation is supported by this repository only on Linux." >&2
  exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Recreate swift-nlg from environment.yml so CUDA Toolkit 12.1 is installed." >&2
  exit 2
fi

export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-build-isolation flash-attn==2.7.3
python -c "import importlib.metadata as m; print('flash-attn', m.version('flash-attn'))"
