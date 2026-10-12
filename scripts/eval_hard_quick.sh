#!/bin/bash
# scripts/eval_hard_quick.sh
# 快速测试：在硬数据集上评测 SFT 模型（限制样本数）

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "🚀 快速 Baseline 测试"
echo "======================"

# 配置
SFT_MODEL="${1:-outputs/sft-short/merged}"
DATA_DIR="${2:-data/train-hard-v2}"
LIMIT="${3:-100}"  # 限制样本数，默认 100 个
BUDGET="${4:-4}"   # 工具调用预算

OUTPUT_DIR="outputs/eval-hard-quick-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📦 配置："
echo "   模型: $SFT_MODEL"
echo "   数据: $DATA_DIR/validation"
echo "   样本数: $LIMIT"
echo "   预算: $BUDGET 次工具调用"
echo "   输出: $OUTPUT_DIR"
echo ""

# 检查必要文件
if [ ! -d "$SFT_MODEL" ]; then
    echo "❌ 模型不存在: $SFT_MODEL"
    exit 1
fi

if [ ! -f "$DATA_DIR/validation/samples.jsonl" ]; then
    echo "❌ 数据不存在: $DATA_DIR/validation/samples.jsonl"
    echo "先生成硬数据集..."
    source .venv/bin/activate
    python -c "
from src.tool_r1.data_hard import generate_hard_dataset
manifest = generate_hard_dataset('$DATA_DIR', train_size=1200, seed=42)
"
fi

echo "▶️  开始评测（这可能需要几分钟加载模型）..."
echo ""

source .venv/bin/activate

python -m tool_r1.eval_suite \
    --data "$DATA_DIR" \
    --sft "$SFT_MODEL" \
    --methods M2 \
    --split validation \
    --budgets "$BUDGET" \
    --limit "$LIMIT" \
    --output "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "======================"
echo "✅ 评测完成！"
echo ""

# 显示结果摘要
if [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "📊 结果摘要："
    python -c "
import json
with open('$OUTPUT_DIR/summary.json') as f:
    data = json.load(f)
    
# 打印关键指标
print()
print('M2 (SFT 模型) 在硬数据集上：')
print()
print(f\"  样本数:     {data.get('num_samples', 'N/A')}\")
print(f\"  正确率:     {data.get('correctness', 'N/A')}\")
print(f\"  平均工具调用: {data.get('avg_calls', 'N/A')}\")
print(f\"  平均奖励:    {data.get('avg_reward', 'N/A')}\")
print()
print('按任务类型分解：')
for task, stats in data.get('by_task', {}).items():
    print(f\"  {task:20s} 正确率: {stats.get('correct', 0):5.1%}  平均调用: {stats.get('avg_calls', 0):.2f}\")
"
fi

echo ""
echo "📁 完整结果保存在: $OUTPUT_DIR/"
echo "   - summary.json      (汇总统计)"
echo "   - trajectories.jsonl (详细轨迹)"
echo "   - report.md        (Markdown 报告)"
