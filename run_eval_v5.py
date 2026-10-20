#!/usr/bin/env python3
"""Run v5 evaluation on GPU 0 using vLLM server + tool_r1 evaluate()."""
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
MODEL_PATH = "/home/lylyly/Tool-R1/Tool-R1/outputs/grpo-hard-v5/merged-step150"
OUTPUT_DIR = "/home/lylyly/Tool-R1/Tool-R1/outputs/eval-v5-test-gpu0"
PORT = 18003
GPU_ID = "0"
SPLIT = "test"
BUDGETS = [1, 2, 3, 4, 6]
SEED = 42
LIMIT = 200  # 200 samples × 5 budgets = 1000 traces ≈ 1-1.5h

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ── Output dir ────────────────────────────────────────────────────────────────
output = Path(OUTPUT_DIR)
if output.exists() and any(f.name != "eval-run.log" for f in output.iterdir()):
    print(f"[WARN] {output} has stale evaluation data (excluding eval-run.log) — remove to avoid stale data", file=sys.stderr)
output.mkdir(parents=True, exist_ok=True)

# ── Check port is free ───────────────────────────────────────────────────────
with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", PORT))
    except OSError:
        print(f"[ERROR] Port {PORT} already in use. Kill the old server first.")
        sys.exit(1)

# ── Start vLLM server ────────────────────────────────────────────────────────
vllm_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--served-model-name", "tool-r1",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--dtype", "bfloat16",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "0.70",
    "--max-model-len", "4096",
    "--max-num-seqs", "8",
    "--enforce-eager",
    "--generation-config", "vllm",
]
server_log = output / "vllm-server.log"
with server_log.open("w", encoding="utf-8") as log_fp:
    server_proc = subprocess.Popen(vllm_cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                                    start_new_session=True)

print(f"[INFO] vLLM server starting on port {PORT} (PID={server_proc.pid})")
print(f"[INFO] Server log: {server_log}")

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
try:
    evaluate(
        data="data",
        output=output / "v5",
        method="M4",   # label only; v5 is similar to M4 method
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
summary_file = output / "v5" / "summary.json"
if summary_file.exists():
    with summary_file.open() as f:
        data = json.load(f)
    overall = data.get("overall", {})
    print("\n" + "="*60)
    print("v5 Evaluation Summary")
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
