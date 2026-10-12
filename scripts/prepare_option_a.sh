#!/bin/bash
# scripts/prepare_option_a.sh
# Quick setup script for Option A: Hard dataset + Aggressive reward GRPO training

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Setting up Option A: Hard Dataset + Aggressive Reward"
echo "=================================================="

# Step 1: Generate hard dataset
echo ""
echo "📊 Step 1: Generating hard dataset (1200 samples)..."
python -c "
from src.tool_r1.data_hard import generate_hard_dataset
from pathlib import Path
import json

output_dir = Path('data/train-hard-v2')
manifest = generate_hard_dataset(output_dir, train_size=1200, seed=42)

print('✓ Hard dataset generated')
print(f\"  Samples: {manifest['splits']['train']['samples']}\")
print(f\"  Documents: {manifest['splits']['train']['documents']}\")
print('Task distribution:')
for task, count in sorted(manifest['splits']['train']['tasks'].items()):
    print(f\"    {task}: {count}\")
"

# Step 2: Show data comparison
echo ""
echo "📈 Step 2: Data comparison"
echo "Original train data:"
ls -lh data/train/samples.jsonl data/train/corpus.jsonl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Hard v2 train data:"
ls -lh data/train-hard-v2/train/samples.jsonl data/train-hard-v2/train/corpus.jsonl | awk '{print "  " $9 " (" $5 ")"}'

# Step 3: Show reward comparison
echo ""
echo "💰 Step 3: Reward functions available"
python -c "
from src.tool_r1.rewards_v2 import score_answer_aggressive, score_answer_extreme, score_answer_cost_only
import json

# Mock prediction and target
pred = {'type': 'number', 'value': 100, 'unit': 'boxes'}
target = {'type': 'number', 'value': 100, 'unit': 'boxes'}

print('Correct answer reward across different call counts:')
print('  Calls | Original (0.1wt) | Aggressive (0.6wt) | Extreme (0.8wt) | Cost-Only')
print('  ------|-----------------|-------------------|-----------------|----------')

from src.tool_r1.rewards import score_answer

for calls in [1, 2, 3, 4, 6]:
    orig = score_answer(pred, target, calls, cost_weight=0.1, cost_reference=6)['reward']
    agg = score_answer_aggressive(pred, target, calls, cost_weight=0.6, cost_reference=6)['reward']
    ext = score_answer_extreme(pred, target, calls, cost_weight=0.8, cost_reference=6)['reward']
    cost = score_answer_cost_only(pred, target, calls, cost_reference=6)['reward']
    print(f'  {calls}    | {orig:.4f}        | {agg:.4f}          | {ext:.4f}       | {cost:.4f}')
"

echo ""
echo "📝 Next steps:"
echo "============"
echo ""
echo "1. Review the hard dataset distribution (should have more multi_hop and complex tasks)"
echo ""
echo "2. Run GRPO training with hard dataset:"
echo "   bash scripts/train_grpo.sh --mode M4-hard-v2 --data_dir data/train-hard-v2 --smoke"
echo ""
echo "3. Or manually launch with specific reward function:"
echo "   python -m tool_r1.training.launch --config configs/train_grpo.yaml \\"
echo "     --overrides 'train_data.dir=data/train-hard-v2/train' 'reward.version=aggressive'"
echo ""
echo "4. Compare final metrics against original M4:"
echo "   - Check pg_loss (should be >0 now, not 0)"
echo "   - Check advantages distribution (should have variance)"
echo "   - Check reward spread across trajectory groups"
echo ""
echo "📖 Documentation: docs/OPTION_A_IMPLEMENTATION.md"
echo ""
