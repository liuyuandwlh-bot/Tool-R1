"""Sequential local model serving and evaluation on a single Linux GPU."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .experiments import evaluate, compare_runs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--sft", default="outputs/sft/merged")
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--output", default="outputs/evaluation-test-seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=[f"M{i}" for i in range(6)], default=[f"M{i}" for i in range(6)])
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Run the serving suite on the Linux GPU server")
    if len(set(args.methods)) != len(args.methods) or args.startup_timeout <= 0:
        raise ValueError("Methods must be unique and startup timeout positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    sft = Path(args.sft).resolve()
    sft_metadata = json.loads((sft / "tool_r1_sft.json").read_text(encoding="utf-8"))
    base = sft_metadata["base_model"]
    revision = sft_metadata.get("base_model_revision")
    if not Path(base).is_dir():
        if not revision:
            raise ValueError("SFT must record a base model revision for fair M0/M1 comparisons")
        from huggingface_hub import snapshot_download
        base = snapshot_download(base, revision=revision)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "suite.json", {"args": vars(args), "base": base, "base_revision": revision,
                                     "status": "running", "model_results": "pending"})
    for method in args.methods:
        source = base if method in {"M0", "M1"} else str(sft) if method == "M2" else str(Path(args.runs).resolve() / f"{method}-seed{args.training_seed}" / "model")
        if not Path(source).is_dir():
            raise FileNotFoundError(f"Missing model: {source}; export its checkpoint before evaluation")
        # Fail before launch if another service is bound; never stop an existing service.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", source,
                   "--served-model-name", "tool-r1", "--host", "127.0.0.1", "--port", str(args.port),
                   "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.70",
                   "--max-model-len", "4096", "--max-num-seqs", "8", "--enforce-eager", "--generation-config", "vllm"]
        write_json(output / f"{method}-server.json", {"command": command})
        with (output / f"{method}-server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{method} server exited; see its server log")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Model serving startup timed out; inspect server log")
                    time.sleep(2)
                print(f"Evaluating {method} ({args.split})", flush=True)
                evaluate(args.data, output / method, method=method, model_source=source,
                         split=args.split, budgets=args.budgets, seed=args.seed, limit=args.limit,
                         base_url=f"http://127.0.0.1:{args.port}/v1",
                         progress=lambda done, total: print(f"{method}: {done}/{total}", flush=True))
            finally:
                # The child was started in a fresh process session; terminate only
                # that session, including its vLLM workers, when this method ends.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    for left, right in (("M2", "M3"), ("M3", "M4"), ("M4", "M5")):
        if left in args.methods and right in args.methods:
            write_json(output / f"{left}-vs-{right}.json", compare_runs(output / left, output / right))
    write_json(output / "suite.json", {"args": vars(args), "base": base, "base_revision": revision, "status": "complete"})


if __name__ == "__main__":
    main()
