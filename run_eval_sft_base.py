#!/usr/bin/env python3
"""Run SFT-base evaluation on GPU 2 using vLLM server + tool_r1 evaluate().

Settings mirror run_eval_v5.py exactly so that v5 GRPO vs SFT-base is a
direct comparison on the same 1000 trajectories.
"""
from __future__ import annotations
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/lylyly/Tool-R1/Tool-R1/outputs/sft-short/merged"
OUTPUT_DIR = "/home/lylyly/Tool-R1/Tool-R1/outputs/eval-sft-base-gpu2"
PORT = 18004
GPU_ID = "2"
SPLIT = "test"
BUDGETS = [1, 2, 3, 4, 6]
SEED = 42
LIMIT = 200  # 200 samples × 5 budgets = 1000 traces (matches v5 eval exactly)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ── Output dir ────────────────────────────────────────────────────────────────
output = Path(OUTPUT_DIR)
output.mkdir(parents=True, exist_ok=True)

# ── Check port is free ───────────────────────────────────────────────────────
with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", PORT))
    except OSError:
        print(f"[ERROR] Port {PORT} already in use. Kill the old server first.")
        sys.exit(1)

# ── Start vLLM server ────────────────────────────────────────────────────────
# GPU 2 total 48GB, ~14.9 GB used by another process → we cap at 30 GB
# 30 / 49 ≈ 0.61
vllm_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--served-model-name", "tool-r1",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--dtype", "bfloat16",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "0.30",
    "--max-model-len", "5120",  # SFT base uses more tool calls than GRPO, can exceed 4096
    "--max-num-seqs", "4",  # reduced to fit longer context in available budget
    "--enforce-eager",
    "--generation-config", "vllm",
]
server_log = output / "vllm-server.log"
with server_log.open("w", encoding="utf-8") as log_fp:
    server_proc = subprocess.Popen(vllm_cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                                    start_new_session=True)

print(f"[INFO] vLLM server starting on port {PORT} (PID={server_proc.pid})")
print(f"[INFO] Server log: {server_log}")
print(f"[INFO] GPU {GPU_ID}, gpu-mem-util=0.61 (~30GB)")

# ── Wait for server to be ready ───────────────────────────────────────────────
deadline = time.monotonic() + 900  # 15 min
while True:
    if server_proc.poll() is not None:
        print("[ERROR] vLLM server exited prematurely. Check log.")
        sys.exit(1)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            if r.status == 200:
                break
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    if time.monotonic() > deadline:
        print("[ERROR] vLLM server startup timed out.")
        sys.exit(1)
    time.sleep(3)

print("[INFO] vLLM server is ready.")

# ── Run evaluation ───────────────────────────────────────────────────────────
from tool_r1.experiments import evaluate

base_url = f"http://127.0.0.1:{PORT}/v1"
model_source = MODEL_PATH

print(f"[INFO] Starting evaluation (split={SPLIT}, budgets={BUDGETS}, limit={LIMIT})")
print(f"[INFO] Method label: M2 (SFT baseline)")
try:
    evaluate(
        data="data",
        output=output / "sft",
        method="M2",   # label: SFT baseline (no GRPO)
        model_source=model_source,
        split=SPLIT,
        budgets=BUDGETS,
        seed=SEED,
        limit=LIMIT,
        base_url=base_url,
        model="tool-r1",
        progress=lambda done, total: print(f"  → {done}/{total}", flush=True),
    )
    print("[INFO] Evaluation complete.")
finally:
    # ── Shutdown vLLM server ────────────────────────────────────────────────
    print("[INFO] Shutting down vLLM server...")
    try:
        os.killpg(server_proc.pid, signal.SIGTERM)
        server_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server_proc.pid, signal.SIGKILL)
        server_proc.wait(timeout=10)
    print("[INFO] Done.")

# ── Print summary ─────────────────────────────────────────────────────────────
summary_file = output / "sft" / "summary.json"
if summary_file.exists():
    with summary_file.open() as f:
        data = json.load(f)
    overall = data.get("overall", {})
    print("\n" + "="*60)
    print("SFT-base Evaluation Summary")
    print("="*60)
    print(f"  Accuracy     : {overall.get('accuracy', 'N/A'):.3f}")
    print(f"  Mean reward  : {overall.get('mean_reward', 'N/A'):.4f}")
    print(f"  Mean cost    : {overall.get('mean_cost', 'N/A'):.4f}")
    print(f"  Mean calls   : {overall.get('mean_tool_calls', 'N/A'):.2f}")
    print(f"  Budget viol. : {overall.get('budget_violation_episode_rate', 'N/A'):.3f}")
    print(f"  Invalid req. : {overall.get('invalid_request_rate', 'N/A'):.4f}")
    print(f"  Exec failures: {overall.get('execution_failure_rate', 'N/A'):.4f}")
    print(f"  Total n      : {overall.get('n', 'N/A')}")
    print("="*60)
    print(f"Full results: {summary_file}")