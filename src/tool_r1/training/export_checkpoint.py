"""Export a selected verl LoRA checkpoint using the exact merged SFT base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from .artifacts import verify_merged_files


def checkpoint_adapter(run_dir: str | Path, step: int | None = None) -> tuple[Path, Path, int]:
    run = Path(run_dir).resolve()
    manifest = json.loads((run / "run.json").read_text())
    if step is None:
        marker = run / "checkpoints/latest_checkpointed_iteration.txt"
        if not marker.exists():
            raise FileNotFoundError("No completed checkpoint marker; inspect training log before exporting")
        step = int(marker.read_text().strip())
    if step < 1:
        raise ValueError("Checkpoint step must be positive")
    adapter = run / "checkpoints" / f"global_step_{step}" / "actor/lora_adapter"
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        if not (adapter / filename).exists():
            raise FileNotFoundError(f"Missing exported adapter: {adapter / filename}; inspect verl save logs")
    model_arg = next(arg for arg in manifest["command"] if arg.startswith("actor_rollout_ref.model.path="))
    base = Path(json.loads(model_arg.split("=", 1)[1]))
    if not (base / "tool_r1_sft.json").exists():
        raise FileNotFoundError("The exact merged SFT base used for GRPO is unavailable")
    if json.loads((base / "tool_r1_sft.json").read_text()) != manifest["sft"]:
        raise ValueError("Merged SFT base metadata changed since this run")
    verify_merged_files(base, manifest["sft"])
    return adapter, base, step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, help="Omit to export the last completed checkpoint, not the best one")
    parser.add_argument("--output", help="Default: <run>/model")
    args = parser.parse_args()
    adapter, base, step = checkpoint_adapter(args.run, args.step)
    output = Path(args.output).resolve() if args.output else Path(args.run).resolve() / "model"
    subprocess.run([sys.executable, "-m", "tool_r1.training.merge", "--base", str(base),
                    "--adapter", str(adapter), "--output", str(output)], check=True)
    (output / "tool_r1_checkpoint.json").write_text(json.dumps({"run": str(Path(args.run).resolve()),
                     "step": step, "selection": "explicit" if args.step else "last_completed"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
