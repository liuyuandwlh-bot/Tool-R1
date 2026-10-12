#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${1:-outputs/sft/merged}"
PORT="${2:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec vllm serve "$MODEL" --served-model-name tool-r1 --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.70 \
  --max-model-len 4096 --max-num-seqs 8 --enforce-eager --generation-config vllm
