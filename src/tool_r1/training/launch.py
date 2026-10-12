"""Build and launch the pinned verl GRPO recipe for one A6000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

VERL_COMMIT = "f9c855f7cf04d603c9546bc01776c74806a879c1"
ROOT = Path(__file__).resolve().parents[3]


def require_resume_checkpoint(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    if not (run / "run.json").is_file():
        raise FileNotFoundError("--resume requires the original run.json")
    marker = run / "checkpoints/latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError("--resume requires a completed checkpoint marker")
    step = int(marker.read_text().strip())
    if step < 1:
        raise ValueError("Invalid resume checkpoint step")
    checkpoint = run / "checkpoints" / f"global_step_{step}"
    required = ["data.pt", "actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
                "actor/extra_state_world_size_1_rank_0.pt", "actor/lora_adapter/adapter_model.safetensors",
                "actor/lora_adapter/adapter_config.json"]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete resume checkpoint: {missing}")
    return checkpoint


def build_overrides(config, *, model, data, prepared, run_dir, mode="M4", seed=42, steps=None, smoke=False):
    if mode not in {"M3", "M4", "M5"}:
        raise ValueError("Mode must be M3 (answer), M4 (cost), or M5 (hidden-budget cost)")
    visible = mode != "M5"
    batch = 2 if smoke else int(config["train_batch_size"])
    values = {
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "data.train_files": str(prepared / f"train_{'visible' if visible else 'hidden'}.parquet"),
        "data.val_files": str(prepared / f"validation_{'visible' if visible else 'hidden'}.parquet"),
        "data.train_batch_size": batch, "data.val_batch_size": batch,
        "data.max_prompt_length": config["prompt_tokens"],
        "data.max_response_length": config["response_tokens_including_observations"],
        "data.return_raw_chat": True, "data.truncation": "error", "data.filter_overlong_prompts": False,
        "data.dataloader_num_workers": 0, "data.seed": seed,
        "data.val_max_samples": 12 if smoke else 96,
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.model.lora_rank": config["lora_rank"],
        "actor_rollout_ref.model.lora_alpha": config["lora_alpha"],
        "actor_rollout_ref.model.target_modules": "all-linear",
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.use_remove_padding": False,
        "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.optim.lr": config["learning_rate"],
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": config["kl_coefficient"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.calculate_entropy": False,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.seed": seed,
        "actor_rollout_ref.actor.data_loader_seed": seed,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": config["group_size"],
        "actor_rollout_ref.rollout.temperature": 0.8, "actor_rollout_ref.rollout.top_p": 0.95,
        "actor_rollout_ref.rollout.gpu_memory_utilization": config["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.max_num_seqs": 4 if smoke else config["max_num_seqs"],
        "actor_rollout_ref.rollout.max_model_len": config["context_tokens"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": config["context_tokens"],
        "actor_rollout_ref.rollout.enforce_eager": True,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed": seed,
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.layered_summon": True,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "tool_r1",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(run_dir / "agent.yaml"),
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "custom_reward_function.path": str(ROOT / "src/tool_r1/integrations/reward.py"),
        "custom_reward_function.name": "compute_score",
        "reward_model.use_reward_loop": False,
        "trainer.nnodes": 1, "trainer.n_gpus_per_node": 1,
        "trainer.use_legacy_worker_impl": "enable",
        "trainer.logger": ["console"], "trainer.project_name": "tool_r1",
        "trainer.experiment_name": run_dir.name,
        "trainer.total_training_steps": steps if steps is not None else (20 if smoke else config["total_steps"]),
        "trainer.total_epochs": 100, "trainer.val_before_train": True,
        "trainer.test_freq": 10 if smoke else 50,
        "trainer.save_freq": 20 if smoke else 100,
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.validation_data_dir": str(run_dir / "validation"),
        "trainer.resume_mode": "disable",
        "ray_kwargs.ray_init.num_cpus": 8,
    }
    # Hydra argument strings are not shell strings; subprocess passes each intact.
    overrides = [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]
    agent = [{"name": "tool_r1", "_target_": "tool_r1.integrations.verl_agent.ToolR1AgentLoop",
              "data_root": str(data), "cost_weight": 0.0 if mode == "M3" else config["cost_weight"],
              "cost_reference": config["cost_reference"], "max_action_tokens": config["action_tokens"],
              "python_backend": "restricted", "trace_dir": str(run_dir / "trajectories")}]
    return overrides, agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["M3", "M4", "M5"], default="M4")
    parser.add_argument("--model", default="outputs/sft/merged")
    parser.add_argument("--data", default="data")
    parser.add_argument("--prepared", default="data/prepared")
    parser.add_argument("--config", default=str(ROOT / "configs/a6000.json"))
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true", help="20-update capacity run, batch=2; not a benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print exact recipe without downloading or launching")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoint in this explicit output directory")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["prompt_tokens"] + config["response_tokens_including_observations"] != config["context_tokens"]:
        raise ValueError("Prompt + response budgets must equal model context")
    run_dir = Path(args.output or f"outputs/{args.mode}-seed{args.seed}{'-smoke' if args.smoke else ''}").resolve()
    model, data, prepared = Path(args.model).resolve(), Path(args.data).resolve(), Path(args.prepared).resolve()
    overrides, agent = build_overrides(config, model=model, data=data, prepared=prepared,
                         run_dir=run_dir, mode=args.mode, seed=args.seed, steps=args.steps, smoke=args.smoke)
    if args.resume:
        checkpoint = require_resume_checkpoint(run_dir)
        overrides = [arg if not arg.startswith("trainer.resume_mode=") else 'trainer.resume_mode="resume_path"' for arg in overrides]
        overrides.append("trainer.resume_from_path=" + json.dumps(str(checkpoint)))
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    if args.dry_run:
        print(json.dumps({"command": command, "agent_config": agent, "verl_commit": VERL_COMMIT}, indent=2, ensure_ascii=False))
        return
    if sys.platform != "linux":
        raise RuntimeError("Run GRPO on the Linux NVIDIA server; use --dry-run locally")
    import importlib.metadata
    # The v0.7.0 Git tag retains "0.7.0.dev" in its version file;
    # packaging normalizes it to 0.7.0.dev0. The commit check is authoritative.
    if importlib.metadata.version("verl") not in {"0.7.0", "0.7.0.dev", "0.7.0.dev0"}:
        raise RuntimeError("Install the pinned verl v0.7.0 server stack")
    direct_url = importlib.metadata.distribution("verl").read_text("direct_url.json")
    if not direct_url or json.loads(direct_url).get("vcs_info", {}).get("commit_id") != VERL_COMMIT:
        raise RuntimeError("verl must be installed from the exact Git commit in requirements-server.txt")
    sft_info = model / "tool_r1_sft.json"
    if not sft_info.exists() or not json.loads(sft_info.read_text()).get("merged"):
        raise ValueError("--model must be a merged Tool-R1 SFT model; see training.merge")
    from .artifacts import verify_merged_files
    verify_merged_files(model, json.loads(sft_info.read_text()))
    from tool_r1.data import audit_dataset
    
    # Allow skipping data audit for hard datasets via SKIP_DATA_AUDIT=1
    skip_audit = os.environ.get("SKIP_DATA_AUDIT") == "1"
    
    if not skip_audit:
        audit = audit_dataset(data)
        prepared_info = json.loads((prepared / "manifest.json").read_text())
        if not audit["ok"] or prepared_info["dataset_sha256"] != audit["dataset_sha256"]:
            raise ValueError("Dataset/export audit mismatch; re-run training.prepare")
        if prepared_info.get("budgets") != config["training_budgets"]:
            raise ValueError("Prepared data budgets differ from the declared training budgets")
        if json.loads(sft_info.read_text()).get("dataset_sha256") != audit["dataset_sha256"]:
            raise ValueError("SFT and GRPO must use the same audited dataset manifest")
        for item in prepared_info["files"]:
            if hashlib.sha256((prepared / item["name"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Prepared parquet was modified; re-export audited data")
        audit_hash = audit["dataset_sha256"]
    else:
        audit_hash = "<hard-dataset-unaudited>"
    
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Run already exists; use --resume or a new --output: {run_dir}")
    manifest = {"args": vars(args), "config": config, "command": command, "agent": agent,
                "verl_commit": VERL_COMMIT, "sft": json.loads(sft_info.read_text()),
                "dataset_sha256": audit_hash, "capacity_test": args.smoke}
    previous = run_dir / "run.json"
    if args.resume:
        old = json.loads(previous.read_text())
        if old["config"] != config or any(old["args"][key] != getattr(args, key) for key in ("mode", "seed", "smoke")) or old["sft"] != manifest["sft"]:
            raise ValueError("Resume config/model differs from original run")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "agent.yaml").write_text(json.dumps(agent, indent=2), encoding="utf-8")
    previous.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = dict(os.environ)
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("VLLM_USE_V1", "1")
    environment.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")
    environment.setdefault("WANDB_MODE", "disabled")
    environment.setdefault("OMP_NUM_THREADS", "4")
    print("Launching:", shlex.join(command), flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
