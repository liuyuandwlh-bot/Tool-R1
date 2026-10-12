# Training integration provenance

The implementation targets **verl v0.7.0**, Git commit
`f9c855f7cf04d603c9546bc01776c74806a879c1`, Python 3.11, Torch 2.8.0,
vLLM 0.11.0, Transformers 4.57.1 and PEFT 0.17.1. All training commands require
the Linux GPU server. No GPU training or measured model benchmark is claimed
from the Windows development checks.

The v0.7.0 Git tag's version file still says `0.7.0.dev` (package metadata
normalizes this to `0.7.0.dev0`). The installer and launcher identify this
release by its exact Git commit, rather than rejecting that version string.

The exact upstream sources inspected during implementation:

- [AgentLoopBase constructor, token masks and reward_score](https://github.com/verl-project/verl/blob/f9c855f7cf04d603c9546bc01776c74806a879c1/verl/experimental/agent_loop/agent_loop.py)
- [Async vLLM server max_tokens and LoRARequest forwarding](https://github.com/verl-project/verl/blob/f9c855f7cf04d603c9546bc01776c74806a879c1/verl/workers/rollout/vllm_rollout/vllm_async_server.py)
- [FSDP disabled-adapter reference policy and lora_adapter checkpoint export](https://github.com/verl-project/verl/blob/f9c855f7cf04d603c9546bc01776c74806a879c1/verl/workers/fsdp_workers.py)
- [verl supported vLLM dependency range](https://github.com/verl-project/verl/blob/f9c855f7cf04d603c9546bc01776c74806a879c1/setup.py)
- [vLLM 0.11.0 CUDA dependencies, including Torch 2.8.0](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)

Version 0.6.0 was not selected: its general LoRA examples do not establish that
the experimental asynchronous AgentLoop server forwards the updated adapter.
Version 0.7.0 explicitly forwards the loaded LoRA request in generation.

## Model and objective invariants

- Supported templates are unmodified **Qwen2.5-Instruct ChatML**. A runtime
  assertion rejects a different template. The supported default is 3B, with
  1.5B as the capacity/debug fallback.
- The training budget values are **2 and 4**. Budget 3 is available for
  interpolation evaluation and 6 for extrapolation evaluation.
- Train and validation are exported independently, with split-specific corpora.
  The held-out test data is never exported to the training framework.
- M3, M4 and M5 load the exact same merged SFT base and fresh RL adapters. M3
  sets cost_weight=0, M4 sets it to 0.1, M5 hides the budget and uses 0.1.
- Merging the SFT adapter before RL is necessary: verl computes the reference
  log probabilities with the RL adapter disabled. The reference must therefore
  be the merged SFT policy, not the original pretrained model.
- Cost is admitted_calls/6, capped at 1. The denominator stays fixed across
  budgets. Incorrect, incomplete or truncated answers receive zero reward.
- Assistant samples retain their original token IDs. Tool observations and
  inserted role headers have response_mask=0. The final reward comes from the
  trusted Environment answer and call counters; it is not regex-extracted from
  tool text. A fallback to legacy transcript-based scoring fails explicitly.

## Resource choices and checks

The initial recipe is rank-32 LoRA, G=4, one generated action up to 384 tokens,
1,024 initial prompt tokens and 3,072 continuation tokens including observations.
The total context ceiling is 4,096. Microbatch size is 1; the rollout engine has
35% GPU memory utilization and eager execution; model/optimizer state is
offloaded between stages. SDPA with remove_padding=False avoids a FlashAttention
source build. This trades throughput for a simpler capacity baseline.

Eight focused CPU/mock checks cover token masks, exact ChatML continuation,
assistant-only SFT labels, truncated-action rejection, trusted rewards,
consistent experiment controls, complete resume state and checkpoint/base matching. The actual Hydra
1.3.2 composer also successfully composed all launcher overrides against the 17
configuration files downloaded from the exact pinned upstream commit. Repeat:

```bash
python scripts/check_verl_config.py
python -m unittest discover -s tests -v
```

These checks do not establish CUDA-kernel compatibility, GPU peak memory,
training throughput or learned model quality. The server acceptance step is:

```bash
bash scripts/train_grpo.sh --mode M4 --smoke
```

The smoke run performs 20 updates and saves real logs/checkpoints. A capacity
failure should be resolved before the 500-update experiment suite. Do not use
smoke-run metrics as the final experiment result. The sequential suite exports
the **last completed** checkpoint; selecting a different checkpoint must use
validation data and an explicit `training.export_checkpoint --step`.

The installer records the full resolved environment in
`outputs/environment/pip-freeze.txt`; transitive packages are not all frozen
in the handwritten requirements file. Reuse this resolved environment across
all comparisons. CUDA 12.8 wheels require a compatible NVIDIA driver (Linux
570.26 or newer for native CUDA 12.8 support).
