# A6000 服务器操作手册

以下命令均在上传后的项目根目录执行，面向 Linux + 单卡 RTX A6000 48GB。未提供服务器连接，本地交付不包含已训练权重。

## 1. 安装与自检

先准备 Python 3.11 和兼容的 NVIDIA 驱动。安装器不会修改系统驱动。

```bash
cd Tool-R1
bash scripts/install_server.sh
source .venv/bin/activate
python -m tool_r1 preflight --require-a6000 --require-stack --output outputs/environment/preflight.json
```

安装器在下载训练依赖前检查硬件，在安装后检查 CUDA、BF16、包版本、verl 固定提交以及 Hydra 配置组合。使用独立 `.venv`，保存 `pip-freeze.txt`。若 Python 名称不同，可用 `PYTHON=/absolute/path/python3.11 bash scripts/install_server.sh`。

主依赖是 Torch 2.8.0、vLLM 0.11.0、Transformers 4.57.1、PEFT 0.17.1；verl 固定到 `f9c855f7cf04d603c9546bc01776c74806a879c1`。需能访问 PyPI、GitHub 和 Hugging Face，初次下载模型需要磁盘与网络。

## 2. 生成并验证数据

```bash
python -m tool_r1 generate
python -m tool_r1 audit
python -m pytest -q
python -m tool_r1 smoke --output outputs/scripted-smoke
python -m tool_r1.training.prepare
```

默认训练/验证/测试题量为 2400/400/800。每个划分有独立语料；示范仅位于训练集。数据导出会再次实际执行所有教师动作，检查 Python 输出、检索证据和最终答案，得到 4800 条逐轮 SFT 样本与四个训练/验证 parquet 文件。测试集不会导入训练 parquet。

重新生成相同配置的数据会得到同一指纹。生成命令会重写目标数据文件；已有实验请先使用 `audit`，调试使用其他 `--data` 目录，避免覆盖。更换数据后应重新 SFT，不能接续旧 GRPO。

## 3. 最低成本试跑

若希望先验证安装和训练闭环，使用较小模型与 10 次 SFT 更新：

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct SFT_OUTPUT=outputs/sft-smoke \
  bash scripts/train_sft.sh --max-steps 10
bash scripts/train_grpo.sh --mode M4 --model outputs/sft-smoke/merged --smoke
python -m tool_r1.training.export_checkpoint --run outputs/M4-seed42-smoke
```

`--smoke` 对 GRPO 设置 20 次更新、训练 batch=2、group=4，并减少验证样本。该结果只用于容量与管线检查。10 次 SFT 不保证协议能力足够，不能据此评估 RL 收益。

正式默认模型的检查流程：

```bash
bash scripts/train_sft.sh
bash scripts/train_grpo.sh --mode M4 --smoke --output outputs/M4-3B-capacity
```

若先执行过小模型示例，正式 3B 容量测试必须使用不同输出目录。两个 smoke 都不参与正式统计。

可在第二个终端记录整张卡的显存占用：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 2 > outputs/gpu-capacity.csv
```

容量测试结束后手工停止监测。该值包括卡上其他进程，应确保 GPU 没有同时运行其他任务。

验收：20 次更新完成；生成、工具执行、反向传播、验证和 checkpoint 保存均成功；日志中的奖励、KL、梯度等为有限值。若所有组的奖励都相同，GRPO 可能没有有效学习信号，需要检查 SFT 饱和度或协议失败；完成更新本身不代表模型已经变好。

## 4. 正式训练与导出

```bash
bash scripts/run_grpo_suite.sh --seed 42
```

M3、M4、M5 顺序运行，每个默认 500 次更新，自动将最后完成的 checkpoint 导出为可服务的合并模型：

```text
outputs/sft/merged
outputs/M3-seed42/model
outputs/M4-seed42/model
outputs/M5-seed42/model
```

默认训练 batch=8、每题 4 条轨迹，共 32 条 rollout/更新；每种方法约 16000 条训练轨迹，三种约 48000 条。训练时间要用容量测试的平均步耗时估算：`正式更新次数 × 稳态秒/更新`，再加验证、保存、SFT 和评测时间。本地没有测过 A6000 吞吐，不承诺训练时长。

输出包括 `train.log`、`run.json`、`agent.yaml`、`trajectories/`、`validation/`、`checkpoints/`。保留原始记录，不依据测试结果反复调参。

## 5. 验证集调试与冻结测试

先对少量验证题确认模型服务正常：

```bash
bash scripts/evaluate_suite.sh --methods M1 M2 M3 M4 M5 --split validation \
  --budgets 2 4 --limit 16 --output outputs/eval-validation-small
```

随后用完整验证集确定配置，固定后评测测试集：

```bash
bash scripts/evaluate_suite.sh --split validation --budgets 2 4 --output outputs/eval-validation
bash scripts/evaluate_suite.sh --split test --budgets 1 2 3 4 6 --output outputs/eval-test
```

默认完整测试需要 20800 条轨迹：M1–M5 各 800×5 条，M0 单独 800 条。可先保留预算 2、4，其他预算写为未完成。单卡会顺序启动各模型；已有服务占用端口时脚本退出，不会结束已有服务。

各方法输出报告与原始轨迹，suite 自动生成 `M2-vs-M3.json`、`M3-vs-M4.json`、`M4-vs-M5.json`。`right_minus_left` 表示右边减左边；`efficiency_supported` 要同时满足正确率差区间下界 ≥ −0.02 和调用差区间上界 < 0。有限样本、不稳定训练和合成数据范围仍需要在结论中说明。

也可手动服务并评测单个模型：

```bash
# 终端 A
bash scripts/serve_model.sh outputs/M4-seed42/model 8000
# 终端 B，激活同一虚拟环境
python -m tool_r1 evaluate --method M4 --model-source outputs/M4-seed42/model \
  --split test --budgets 2 4 --output outputs/M4-test
```

手工模式中，操作者负责确保服务实际加载的权重与 `--model-source` 一致。模型别名默认 `tool-r1`。HTTP API Key 如需使用，通过 `TOOL_R1_API_KEY` 环境变量配置；不要放进 URL 或日志。推荐顺序 suite，服务命令和模型路径会一起保存。

## 6. 续训与多种子

```bash
bash scripts/train_grpo.sh --mode M4 --output outputs/M4-seed42 --resume
```

续训需要原始运行清单、完整 checkpoint、相同 SFT 与数据。缺文件会明确失败，不会自动从头开始。迁移已有实验目录时需保留记录中的模型路径，或明确重新导出和验证，不能直接改清单掩盖路径变化。

复现训练波动时，从同一 SFT 初始化运行不同 RL seed：

```bash
bash scripts/run_grpo_suite.sh --seed 43
bash scripts/run_grpo_suite.sh --seed 44
```

分别用 `--training-seed 43` 或 `44` 评测，并保留相同评测 seed 42。跨种子汇总需单独分析；bootstrap 区间只反映评测题目抽样误差。

## 7. 常见故障

| 现象 | 处理 |
|---|---|
| 驱动或 Python 自检失败 | 按要求准备系统环境；不要混装另一套 vLLM/Torch 绕过检查 |
| 下载失败 | 检查服务器网络和镜像源，保留包的固定版本与 verl commit |
| 显存不足 | 先确认无其他 GPU 进程；将 `max_num_seqs` 从 8 降到 4、训练 batch 从 8 降到 4；仍不足时改 1.5B 并重做所有对照 |
| 上下文超长 | 检查查询/代码是否重复或输出过长；不能无声截断 final 后给奖励 |
| 生成服务错误 | 查看对应 server log；评测保留部分轨迹并标记失败，用新目录重跑 |
| 原模型校验失败 | 检查是否修改或移动了权重，使用正确导出入口，不要删除校验 |
| 奖励没有变化 | 检查输出协议、答案正确率和同组轨迹差异；可能是全错或全对且成本相同 |
| Docker 不可用 | 使用默认 `restricted` 后端；Docker 只作为可选隔离执行方案 |

显存调整会改变实验配置，应同时应用于 M3/M4/M5 并重新开始对应实验。group size 不建议降到 1，GRPO 需要组内相对奖励信号。不要把 1.5B 与 3B 的结果当作仅改变奖励的消融。

## 8. 完整自动流程

安装完成、容量测试通过且尚未正式生成 SFT/实验目录时，可以执行：

```bash
bash scripts/run_pipeline.sh
```

该脚本生成/审计数据、测试、准备数据、SFT、三组 GRPO、验证和测试。已经做过 SFT 时从第 4 步继续，避免重复输出冲突。
