#!/bin/bash
# scripts/eval_hard_baseline.sh
# 测试现有模型在硬数据集上的 baseline 性能

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "🧪 Baseline 测试：现有模型在硬数据集上的性能"
echo "======================================================"

# 使用原 SFT 权重
SFT_MODEL="${1:-outputs/sft-short/merged}"
DATA_DIR="${2:-data/train-hard-v2}"

echo "📦 配置："
echo "   模型: $SFT_MODEL"
echo "   数据: $DATA_DIR"
echo ""

# 检查模型是否存在
if [ ! -d "$SFT_MODEL" ]; then
    echo "❌ 错误：模型目录不存在: $SFT_MODEL"
    echo "可用的模型："
    ls -d outputs/*/merged 2>/dev/null || ls -d outputs/sft*/merged 2>/dev/null
    exit 1
fi

# 检查数据是否存在
if [ ! -d "$DATA_DIR/train" ]; then
    echo "❌ 错误：数据目录不存在: $DATA_DIR/train"
    echo "生成硬数据集..."
    source .venv/bin/activate
    python -c "
from src.tool_r1.data_hard import generate_hard_dataset
manifest = generate_hard_dataset('$DATA_DIR', train_size=1200, seed=42)
print(f'✓ 硬数据集已生成: {manifest[\"splits\"][\"train\"][\"samples\"]} 样本')
"
fi

# 创建输出目录
OUTPUT_DIR="outputs/eval-hard-baseline-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📊 运行评测..."
echo "   输出目录: $OUTPUT_DIR"
echo ""

# 运行评测（使用 M2 = SFT 模型）
source .venv/bin/activate

python -m tool_r1.eval_suite \
    --data "$DATA_DIR" \
    --sft "$SFT_MODEL" \
    --methods M2 \
    --split train \
    --budgets 2 4 \
    --limit 200 \
    --output "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "======================================================"
echo "✅ 评测完成！"
echo ""
echo "📈 查看结果："
echo "   cat $OUTPUT_DIR/summary.json"
echo ""
echo "📊 详细轨迹："
echo "   head -50 $OUTPUT_DIR/trajectories.jsonl"
echo ""
echo "📋 报告："
if [ -f "$OUTPUT_DIR/report.md" ]; then
    cat "$OUTPUT_DIR/report.md"
fi
