#!/usr/bin/env bash
# Full sequential experiment. First complete the capacity check in the runbook.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
python -m tool_r1 preflight --require-a6000 --require-stack --output outputs/environment/preflight.json
if [[ ! -e data/manifest.json ]]; then
  python -m tool_r1 generate
fi
python -m tool_r1 audit
python -m pytest -q
python -m tool_r1.training.prepare
bash scripts/train_sft.sh --seed "$SEED"
bash scripts/run_grpo_suite.sh --seed "$SEED"
bash scripts/evaluate_suite.sh --training-seed "$SEED" --seed 42 --split validation --budgets 2 4 --output "outputs/eval-validation-trainseed$SEED"
bash scripts/evaluate_suite.sh --training-seed "$SEED" --seed 42 --split test --budgets 1 2 3 4 6 --output "outputs/eval-test-trainseed$SEED"
