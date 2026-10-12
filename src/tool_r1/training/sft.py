"""Single-GPU LoRA SFT over replayed, assistant-only training turns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tool_r1.training.chat import sft_turn, validate_chatml
from tool_r1.training.prepare import replay_demonstrations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/sft/adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    args = parser.parse_args()
    from tool_r1.data import audit_dataset
    audit = audit_dataset(args.data)
    if not audit["ok"]:
        raise ValueError(f"Dataset audit failed: {audit['errors']}")
    # Replay again at the point of use; a stale prepared file cannot bypass checks.
    turns = replay_demonstrations(args.data)
    if not turns:
        raise ValueError("No train-only SFT demonstrations")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires the Linux CUDA server with BF16 support")
    set_seed(args.seed)
    local_model = Path(args.model).is_dir()
    model_ref = str(Path(args.model).resolve()) if local_model else args.model
    model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", trust_remote_code=False)
    base_revision = getattr(model.config, "_commit_hash", None)
    if not local_model and not base_revision:
        raise ValueError("The remote base model did not report an immutable commit revision")
    # Resolve the tokenizer from exactly the model revision, even if main moves
    # between the two downloads. The same revision is enforced during merging.
    revision_kwargs = {} if local_model else {"revision": base_revision}
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=False, **revision_kwargs)
    validate_chatml(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list([sft_turn(tokenizer, row["messages"], row["action"], args.max_length) for row in turns])
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                         target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def collate(rows):
        length = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in rows], dtype=torch.long)
                for key, padding in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}

    output = Path(args.output).resolve()
    if (output / "adapter_model.safetensors").exists():
        raise FileExistsError(f"Adapter already exists; choose a fresh --output: {output}")
    training_args = TrainingArguments(output_dir=str(output), num_train_epochs=args.epochs,
            max_steps=args.max_steps, per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
            warmup_ratio=0.05, weight_decay=0.0, lr_scheduler_type="cosine", bf16=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, save_strategy="steps", save_steps=100, save_total_limit=2,
            report_to=[], dataloader_num_workers=0, seed=args.seed, data_seed=args.seed,
            remove_unused_columns=False, optim="adamw_torch", max_grad_norm=1.0)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    manifest = {"stage": "sft", "base_model": model_ref, "seed": args.seed,
                "dataset_sha256": audit["dataset_sha256"], "training_examples": len(dataset),
                "demonstrations_replayed": True, "assistant_only_loss": True,
                "configuration": vars(args), "metrics": result.metrics,
                "base_model_revision": base_revision,
                "adapter_sha256": hashlib.sha256((output / "adapter_model.safetensors").read_bytes()).hexdigest()}
    (output / "tool_r1_sft.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved SFT adapter: {output}. Merge it before GRPO so the KL reference is the SFT policy.")


if __name__ == "__main__":
    main()
