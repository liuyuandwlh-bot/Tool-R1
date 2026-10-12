#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUT="${SFT_OUTPUT:-outputs/sft}"
for arg in "$@"; do
  case "$arg" in
    --model|--model=*|--output|--output=*)
      echo "Set MODEL and SFT_OUTPUT environment variables for this wrapper so training and merge share paths." >&2
      exit 2
      ;;
  esac
done
python -m tool_r1.training.sft --model "$MODEL" --output "$OUT/adapter" "$@"
python -m tool_r1.training.merge --base "$MODEL" --adapter "$OUT/adapter" --output "$OUT/merged"
