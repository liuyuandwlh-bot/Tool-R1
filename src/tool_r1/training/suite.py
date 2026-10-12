"""Run M3/M4/M5 sequentially on one card, then export each final checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/sft/merged")
    parser.add_argument("--data", default="data")
    parser.add_argument("--prepared", default="data/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for mode in ("M3", "M4", "M5"):
        run = Path(args.output_root) / f"{mode}-seed{args.seed}{'-smoke' if args.smoke else ''}"
        command = [sys.executable, "-m", "tool_r1.training.launch", "--mode", mode,
                   "--model", args.model, "--data", args.data, "--prepared", args.prepared,
                   "--seed", str(args.seed), "--output", str(run)]
        if args.steps is not None:
            command.extend(["--steps", str(args.steps)])
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, "-m", "tool_r1.training.export_checkpoint", "--run", str(run)], check=True)


if __name__ == "__main__":
    main()
