#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Use this installer on the Linux GPU server." >&2
  exit 1
fi
PYTHON="${PYTHON:-python3.11}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11"'
nvidia-smi
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m tool_r1.preflight --require-a6000 --workspace "$ROOT"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
.venv/bin/python -m pip install -r configs/requirements-server.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
mkdir -p outputs/environment
.venv/bin/python -m pip freeze > outputs/environment/pip-freeze.txt
.venv/bin/python -c 'import torch, vllm, verl; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print("vLLM",vllm.__version__,"verl",verl.__version__)'
.venv/bin/python -m tool_r1.preflight --require-a6000 --require-stack --output outputs/environment/preflight.json
.venv/bin/python scripts/check_verl_config.py
echo "Installed. Activate: source .venv/bin/activate"
echo "The stock Torch/vLLM wheels use CUDA 12.8; provision an NVIDIA driver supporting CUDA 12.8 (570.26+ on Linux)."
