# Tool-R1

面向 **单张 NVIDIA RTX A6000 48GB** 的工具推理与 GRPO 实验项目。模型逐轮选择直接回答、检索或 Python 计算；训练目标是在答对的前提下减少无效工具调用。

默认路线：**Qwen2.5-3B-Instruct → LoRA SFT → verl GRPO → 同题同预算评测**。

**当前交付的是可运行的工程代码、数据流程和服务器实验流程。GPU 训练与模型效果尚未实测。** 本地教师脚本通过率不能作为模型正确率、RL 提升或简历中的实验收益。

## 从这里开始

- [A6000 服务器操作手册](docs/SERVER_RUNBOOK.md)：上传后按顺序执行。
- [方案与实现取舍](docs/DESIGN.md)：任务、奖励、训练、消融和资源门槛。
- [数据说明](docs/DATA_CARD.md)：数据生成、划分及适用范围。
- [验收状态](docs/VALIDATION.md)：已经验证与仍需服务器验证的内容。
- [上游接口核对](configs/UPSTREAM_API.md)：固定 verl 版本与兼容理由。

## 资源要求

| 项目 | 本项目配置 |
|---|---|
| GPU | 单张 RTX A6000，48GB 显存 |
| 系统 | Linux x86_64，建议 Ubuntu 22.04/24.04 |
| Python | 服务器使用 3.11 |
| NVIDIA 驱动 | 本安装方案要求 570.26 或更新，匹配 CUDA 12.8 轮子 |
| 内存 | 32GiB 为自检下限，建议 64GiB 或以上，需给 CPU offload 留空间 |
| 磁盘 | 短测试至少 60GiB 空闲；完整对照实验建议 150GiB 或以上 |
| CPU | 建议 8 个可用线程或以上 |

A6000 适合从 3B + LoRA 开始。显存是否足够仍由 **20 次更新的容量测试**确认。7B、全参数训练和更长上下文未在本项目验证。默认检索与受限 Python 不需要 Docker、搜索 API 或付费模型 API。

## 上传后快速启动

在服务器解压项目并进入 `Tool-R1` 目录：

```bash
bash scripts/install_server.sh
source .venv/bin/activate
python -m tool_r1 generate
python -m tool_r1 audit
python -m pytest -q
python -m tool_r1.training.prepare
bash scripts/train_sft.sh
bash scripts/train_grpo.sh --mode M4 --smoke
```

容量测试通过后执行正式实验，所有方法从同一份 SFT 权重出发：

```bash
bash scripts/run_grpo_suite.sh --seed 42
bash scripts/evaluate_suite.sh --split validation --budgets 2 4 --output outputs/eval-validation
bash scripts/evaluate_suite.sh --split test --budgets 1 2 3 4 6 --output outputs/eval-test
```

`evaluate_suite.sh` 在一张卡上顺序启动模型服务、评测 M0–M5、释放对应服务并计算配对统计。不要同时启动训练或手工服务。每次运行使用新输出目录，避免覆盖旧实验。

## 无 GPU 验证工程链路

```bash
python -m pip install -e '.[test]'
python -m tool_r1 generate
python -m tool_r1 audit
python -m tool_r1 smoke --output outputs/scripted-smoke
python -m pytest -q
```

核心运行器只用 Python 标准库，适用 Windows/Linux、Python 3.10+。`smoke` 回放训练集教师动作，验证协议、检索证据、计算和奖励；不下载模型。`training.prepare` 额外需要 `pyarrow`，服务器安装器已包含。

## 实验方法

| 方法 | 策略 | 预算输入 |
|---|---|---|
| M0 | 未训练基础模型，无工具，仅诊断 | 无工具 |
| M1 | 未训练基础模型 + 工具协议提示 | 可见 |
| M2 | 工具轨迹 LoRA SFT | 可见 |
| M3 | 同一 SFT 基座 + 正确性 GRPO | 可见 |
| M4 | 同一 SFT 基座 + 正确性与调用成本 GRPO | 可见 |
| M5 | 同一 SFT 基座 + M4 奖励 | 隐藏显式预算 |

训练预算为 `{2,4}`，完整测试预算为 `{1,2,3,4,6}`。M0 无权访问测试新事实，不能用于证明工具效率提升。评测默认贪心解码、每轮最多 384 个输出 token，所有方法使用相同评分函数。

主奖励为 `correct × (1 − 0.1 × admitted_calls / 6)`。工具观察不参与 actor loss 和 KL token loss。结果按任务、预算、措辞模板分组，保留原始轨迹，支持按题聚类的配对 bootstrap。默认正确率退化门槛为 2 个百分点；成本下降且正确率区间满足门槛，才报告效率证据。

## 目录

```text
configs/                   A6000 配置、固定训练依赖、上游核对记录
scripts/                   安装、训练、导出检查、顺序评测、打包入口
src/tool_r1/
  protocol.py, runner.py   JSON 动作协议、预算环境与轨迹
  tools/                  BM25、受限 Python、可选 Docker Python
  data.py, rewards.py      合成数据与严格答案校验
  evaluation.py           指标与配对统计
  experiments.py          评测、溯源、报告
  integrations/           verl AgentLoop 和可信执行奖励
  training/               数据导出、SFT、GRPO、模型合并与恢复
tests/                     无 GPU 单元与集成测试
docs/                      中文文档
data/                      运行时生成，不进入源码仓库
outputs/                   运行日志、模型和评测结果
```

执行 `python -m tool_r1 --help` 查看所有入口。真实评测报告生成于相应输出目录的 `report.md`，详细指标在 `summary.json`，证据在 `traces.jsonl` 和 `manifest.json`。

## 已明确的边界

当前数据是英文合成任务，覆盖直接回答、检索、计算和组合；未包含真实公开 QA 数据、在线搜索或开放式代码任务。隐藏预算条件仍可能通过历史调用和拒绝反馈推断资源状态。Docker 实际容器测试在本地不可用；默认受限解释器已测试。跨三个训练种子的稳定性、真实领域泛化和相对基线提升，需要服务器实验后才能下结论。

源码采用 MIT 许可。verl、模型与第三方依赖使用各自许可证；使用模型前查看所下载版本的模型卡和许可。
