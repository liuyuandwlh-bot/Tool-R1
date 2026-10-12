# 选项 A 实施方案：更难任务 + 激进奖励

## 📋 完成的工作

### 1. 新硬数据集模块 (`src/tool_r1/data_hard.py`)

**新任务类型**（共 1200 个样本推荐规模）：
- `multi_hop_4`（19%）：4 跳链 — company → supplier → region → market → exchange_rate
  - 需要 4 次精准的检索调用
  - 4 个干扰项文档进一步增加难度
  
- `multi_hop_5`（19%）：5 跳链 — multi_hop_4 + 与历史平均值比对
  - 需要 5 次检索调用
  - 需要额外的计算步骤
  
- `chain_of_tools`（19%）：混合工具链 — 检索 → 计算 → 条件过滤 → 最终答案
  - 结合多种工具类型
  - 需要 2 次检索 + 1 次 Python 计算

**原有任务保留**（60% 混合）以保持稳定性。

**使用方法**：
```python
from src.tool_r1.data_hard import generate_hard_dataset
manifest = generate_hard_dataset(output_dir="data/train-hard-v2", train_size=1200, seed=42)
```

### 2. 改进的奖励函数 (`src/tool_r1/rewards_v2.py`)

引入三个激进的奖励设计，都针对**已答对的情况**强烈惩罚成本：

#### A. `score_answer_aggressive` — 推荐用于初期
```
reward = 1.0 - 0.6 * (cost)^1.5  if correct else 0
范围：[0.2, 1.0]（相比原来的 [0.9, 1.0]）
梯度：二次方惩罚，成本越高越贵
```
- 适合：hard 任务初次训练
- 优点：平衡激进性和稳定性

#### B. `score_answer_extreme` — 最激进
```
reward = 1.0 - 0.8 * cost  if correct else 0
范围：[0.2, 1.0]
梯度：线性，每个多余调用 ≈ -0.133 奖励
```
- 适合：correctness 已饱和（>99%）
- 优点：清晰的梯度信号

#### C. `score_answer_cost_only` — 纯成本优化
```
reward = 1.0 - cost  if correct else 0
范围：[0, 1.0]（成本满分则 0 奖励）
梯度：每个调用 = -1/6 奖励
```
- 适合：最后阶段微调
- 优点：最强硬的成本压力

### 3. 数据生成验证

测试输出（200 个样本）：
```
multi_hop_4:    19
multi_hop_5:    19
chain_of_tools: 19
（+其他原有任务）
总文档数：1019（相比原来的样本更多文档干扰项）
```

---

## 🎯 下一步：集成到训练流程

### 步骤 1：修改训练配置

编辑 `configs/` 中的训练配置（或创建新的 YAML）：

```yaml
# configs/train_grpo_hard_v2.yaml
dataset:
  train_dir: "data/train-hard-v2/train"  # 使用新数据集
  validation_dir: "data/validation"      # 保持原有验证集

reward:
  type: "aggressive"  # 或 "extreme", "cost_only"
  cost_weight: 0.6
  cost_reference: 6

training:
  num_epochs: 1
  batch_size: 8
  learning_rate: 1e-5
```

### 步骤 2：修改奖励集成

编辑 `src/tool_r1/integrations/reward.py`，支持多个奖励版本：

```python
def score_episode(answer, ground_truth, metrics, *, 
                  reward_version="aggressive",  # 新参数
                  cost_weight=0.1, 
                  cost_reference=6, 
                  truncated=False):
    
    if reward_version == "aggressive":
        from .rewards_v2 import score_answer_aggressive
        result = score_answer_aggressive(...)
    elif reward_version == "extreme":
        from .rewards_v2 import score_answer_extreme
        result = score_answer_extreme(...)
    else:
        from .rewards import score_answer
        result = score_answer(...)
    
    return {...}
```

### 步骤 3：生成训练数据集

```bash
cd Tool-R1
python -c "
from src.tool_r1.data_hard import generate_hard_dataset
manifest = generate_hard_dataset('data/train-hard-v2', train_size=1200, seed=42)
print('✓ Hard dataset generated')
"
```

### 步骤 4：启动硬数据集 GRPO 训练

```bash
bash scripts/train_grpo.sh \
  --mode M4-hard-v2 \
  --data_dir data/train-hard-v2 \
  --reward_version aggressive
```

---

## 📊 预期效果对比

| 指标 | 原 M4（软数据 + 0.1权重） | 新 M4-hard（硬数据 + aggressive） |
|------|---------------------------|--------------------------------------|
| 训练样本数 | 2400（简单） | 1200（困难） |
| 答对率 baseline | ~98% | ~70-80%（更难） |
| 平均工具调用 | 1.09 | ~2.5-4.0（需要多跳） |
| Reward variance | ~0（无梯度） | 高（有学习信号） |
| Advantages 分布 | 全 0 | [-0.2, +0.2]（有分化） |
| pg_loss | 0（无更新） | >0.01（有梯度） |
| 成本差异 | reward 0.97 vs 0.95 | reward 0.2 vs 1.0（清晰） |

**核心改进**：通过 3-4 倍的 gradient signal variance，使 GRPO 能够学习。

---

##  故障排除

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| 太多答错 | correctness < 50% | 降低难度，混合更多原有任务（改 HARD_TASK_PLAN） |
| 还是没有梯度 | pg_loss 仍为 0 | 检查奖励函数是否正确集成；尝试 `extreme` 版本 |
| VRAM 爆表 | OOM 错误 | 减少 batch_size 或样本数 |
| 答对率掉得快 | correctness 从 80% → 30% | 增加 KL 系数或降低学习率 |

---

## 📚 关键文件

- 新数据生成：`src/tool_r1/data_hard.py`
- 改进奖励：`src/tool_r1/rewards_v2.py`
- 集成入口：`src/tool_r1/integrations/reward.py`（待修改）
- 训练脚本：`scripts/train_grpo.sh`（待修改）

---

## ✅ 完成清单

- [x] 实现 `multi_hop_4/5` 和 `chain_of_tools` 任务生成器
- [x] 生成并验证硬数据集（200 样本测试）
- [x] 实现三个激进奖励函数
- [ ] 修改训练脚本支持新数据集和奖励
- [ ] 运行完整的硬数据集 + aggressive 奖励训练
- [ ] 对比 M4-hard-v2 vs M4 的最终指标
