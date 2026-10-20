#!/usr/bin/env python3
"""Direct evaluation of grpo-hard-v5 merged checkpoint on test set."""
import sys, os, time, socket
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/lylyly/Tool-R1/Tool-R1/outputs/grpo-hard-v5/merged-step150"
DATA_PATH  = "/home/lylyly/Tool-R1/Tool-R1/data"
OUTPUT_DIR = "/home/lylyly/Tool-R1/Tool-R1/outputs/grpo-hard-v5/eval-test-step150"
PORT       = 8008
LIMIT      = 32          # 先跑 32 样本试水；改为 None 则跑全量 800
BUDGETS    = [2, 4, 6]   # tool_call budget 数量
SEED       = 42
GPU_MEM    = 0.50         # 只占 GPU 2 的一半显存
MAX_LEN    = 4096
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # 只用 GPU 2

from tool_r1.experiments import evaluate

# Check port free
with socket.socket() as s:
    try:
        s.bind(("127.0.0.1", PORT))
    except OSError:
        print(f"❌ Port {PORT} is already in use! Kill existing vLLM first.")
        sys.exit(1)

output = Path(OUTPUT_DIR)
if output.exists() and any(output.iterdir()):
    print(f"❌ Output dir exists and non-empty: {output}")
    sys.exit(1)
output.mkdir(parents=True, exist_ok=True)

base_url = f"http://127.0.0.1:{PORT}/v1"

print(f"🚀 Starting vLLM server on port {PORT}")
print(f"   Model: {MODEL_PATH}")
print(f"   GPU mem util: {GPU_MEM}, max_model_len: {MAX_LEN}")
print(f"   Data: {DATA_PATH}, split: test, limit: {LIMIT}")
print(f"   Output: {OUTPUT_DIR}")
print()

# Start vLLM in background
import subprocess, signal, json

vllm_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--served-model-name", "tool-r1",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--dtype", "bfloat16",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", str(GPU_MEM),
    "--max-model-len", str(MAX_LEN),
    "--max-num-seqs", "8",
    "--enforce-eager",
    # NOTE: generation_config.json lives inside MODEL_PATH, vLLM auto-detects it
]
log_file = open(output / "vllm.log", "w")
vllm_proc = subprocess.Popen(vllm_cmd, stdout=log_file, stderr=subprocess.STDOUT,
                              start_new_session=True)

try:
    # Wait for vLLM health
    import urllib.request, urllib.error
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if vllm_proc.poll() is not None:
            print("❌ vLLM server crashed! See vllm.log")
            sys.exit(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                break
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(5)
        print("   ⏳ Waiting for vLLM to be ready...")
    else:
        print("❌ vLLM startup timeout!")
        sys.exit(1)

    print("✅ vLLM ready! Starting evaluation...")
    print()

    # Run evaluation
    evaluate(
        data=DATA_PATH,
        output=output / "M5",
        method="M5",
        model_source=MODEL_PATH,
        split="test",
        budgets=BUDGETS,
        seed=SEED,
        limit=LIMIT,
        base_url=base_url,
        progress=lambda done, total: print(f"   📊 {done}/{total}", flush=True),
    )

    # Print summary
    summary_path = output / "M5" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        overall = summary.get("overall", {})
        print()
        print("=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"   Evaluated:   {summary.get('episodes', '?')} samples")
        print(f"   Budgets:     {BUDGETS}")
        for key in ["reward/mean", "correctness/mean", "cost/mean", "score/mean"]:
            val = overall.get(key)
            if val is not None:
                print(f"   {key:<20}: {val:.4f}")
        print("=" * 60)
        print(f"   Full results: {summary_path}")
    else:
        print("❌ No summary.json found!")

finally:
    print()
    print("🧹 Stopping vLLM...")
    try:
        os.killpg(vllm_proc.pid, signal.SIGTERM)
        vllm_proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(vllm_proc.pid, signal.SIGKILL)
        except: pass
    log_file.close()
    print("✅ Done.")
