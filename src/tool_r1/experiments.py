"""Frozen evaluation records, reproducible scoring, and evidence-only reports."""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import audit_dataset, load_jsonl
from .evaluation import paired_bootstrap, summarize
from .policies import HTTPPolicy, ReplayPolicy
from .rewards import score_answer
from .runner import PROTOCOL_VERSION, run_episode
from .tools.retrieval import BM25


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_digest():
    root = Path(__file__).parent
    files = {str(path.relative_to(root)).replace("\\", "/"): digest(path) for path in sorted(root.rglob("*.py"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _audit(data):
    result = audit_dataset(data)
    if not result["ok"]:
        raise ValueError(f"Dataset audit failed: {result['errors']}")
    return result


def _new_output(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_results(output, traces, manifest):
    summary = summarize(traces)
    templates = sorted({trace["template_family"] for trace in traces if "template_family" in trace})
    summary["per_template_family"] = {key: summarize(t for t in traces if t.get("template_family") == key)["overall"] for key in templates}
    manifest.update(status="complete", episodes=len(traces), traces_sha256=digest(output / "traces.jsonl"))
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summary)
    write_report(output / "report.md", manifest, summary)
    return {"output": str(output), "evaluation_kind": manifest["evaluation_kind"], "overall": summary["overall"]}


def _manifest(audit, *, kind, method, split, budgets, seed, backend):
    return {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_kind": kind, "method": method, "split": split, "budgets": list(budgets),
            "evaluation_seed": seed, "python_backend": backend,
            "dataset_sha256": audit["dataset_sha256"], "code_sha256": code_digest(),
            "protocol_version": PROTOCOL_VERSION, "python": sys.version, "platform": platform.platform(),
            "scoring": {"cost_weight": 0.1, "cost_reference": 6},
            "limitation": "Synthetic curriculum only; no external-domain claims."}


def scripted_smoke(data="data", output="outputs/scripted-smoke", limit=None):
    """Replay train-only teacher actions. This is not a learned policy baseline."""
    data = Path(data)
    audit = _audit(data)
    samples = {row["id"]: row for row in load_jsonl(data / "train/samples.jsonl")}
    demonstrations = load_jsonl(data / "train/demonstrations.jsonl")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        demonstrations = demonstrations[:limit]
    index = BM25(load_jsonl(data / "train/corpus.jsonl"))
    output = _new_output(output)
    manifest = _manifest(audit, kind="scripted_smoke", method="teacher_replay", split="train",
                         budgets=[2, 4], seed=42, backend="restricted")
    write_json(output / "manifest.json", manifest)
    traces = []
    with (output / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for demo in demonstrations:
            sample = samples[demo["sample_id"]]
            budget = (2, 4)[int(hashlib.sha256(sample["id"].encode()).hexdigest(), 16) % 2]
            trace = run_episode(sample, ReplayPolicy(demo["reference_actions"]), index, budget=budget)
            trace["evaluation_kind"] = "scripted_smoke"
            trace["score"] = score_answer(trace["answer"], sample["answer"], trace["metrics"]["admitted_calls"])
            observations = "\n".join(step.get("observation") or "" for step in trace["steps"])
            trace["evidence_covered"] = all(doc_id in observations for doc_id in sample["source_doc_ids"])
            stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
            traces.append(trace)
    result = _save_results(output, traces, manifest)
    result["ok"] = bool(traces) and all(t["score"]["correctness"] and t["evidence_covered"] and
        not any(t["metrics"][key] for key in ("invalid_requests", "execution_failures", "budget_violations")) for t in traces)
    return result


NO_TOOL_SYSTEM = ('Solve the question using your own knowledge and arithmetic. No tools are available. '
                 'Reply with exactly one JSON object and no markdown: '
                 '{"action":"final","answer":{"type":"number","value":6,"unit":"boxes"}}. '
                 'Use the unit requested in the question. Percent value 15 means 15%, not 0.15.')


def _training_control(run, checkpoint):
    """Keep all effective learning settings; omit only treatment and artifact paths."""
    for field, expected_type in (("args", dict), ("config", dict), ("command", list), ("agent", list)):
        if not isinstance(run.get(field), expected_type) or not run[field]:
            raise ValueError(f"Training provenance requires nonempty {field}")
    if not run.get("verl_commit") or not run.get("dataset_sha256"):
        raise ValueError("Training provenance requires verl commit and dataset fingerprints")
    step = checkpoint.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("Exported checkpoint step must be a positive integer")
    omitted_args = {"mode", "model", "data", "prepared", "config", "output", "dry_run", "resume"}
    omitted_overrides = {
        "data.train_files", "data.val_files", "actor_rollout_ref.model.path",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path", "custom_reward_function.path",
        "trainer.experiment_name", "trainer.default_local_dir", "trainer.validation_data_dir",
        "trainer.resume_mode", "trainer.resume_from_path",
    }
    try:
        override_start = run["command"].index("verl.trainer.main_ppo") + 1
    except ValueError as exc:
        raise ValueError("Training command does not identify the expected verl entrypoint") from exc
    overrides, seen = {}, set()
    for item in run["command"][override_start:]:
        if not isinstance(item, str):
            raise ValueError("Training command must contain strings")
        if "=" not in item:
            continue
        key, encoded = item.split("=", 1)
        if key in seen:
            raise ValueError(f"Duplicate training override: {key}")
        seen.add(key)
        if key not in omitted_overrides:
            try:
                overrides[key] = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Training override is not JSON encoded: {key}") from exc
    if not overrides:
        raise ValueError("Training provenance has no effective overrides")
    agent = []
    expected_cost_weight = 0.0 if run["args"].get("mode") == "M3" else run["config"].get("cost_weight")
    for specification in run["agent"]:
        if not isinstance(specification, dict):
            raise ValueError("Training agent configuration must be a dictionary")
        if expected_cost_weight is None or specification.get("cost_weight") != expected_cost_weight:
            raise ValueError("Training reward weight differs from its declared method/config")
        # Reward weight is the M3/M4 treatment. Config still fixes its nonzero
        # value so M4/M5 must agree; paths carry no learning hyperparameters.
        agent.append({key: value for key, value in specification.items()
                      if key not in {"cost_weight", "data_root", "trace_dir"}})
    return {"config": run["config"], "args": {key: value for key, value in run["args"].items() if key not in omitted_args},
            "effective_overrides": overrides, "agent": agent, "checkpoint_step": step,
            "verl_commit": run["verl_commit"], "dataset_sha256": run["dataset_sha256"]}


def _model_provenance(source, method):
    path = Path(source)
    result = {"declared_source": str(source),
              "binding": "Operator must serve these weights under the declared endpoint alias."}
    if path.is_dir():
        from .training.artifacts import verify_merged_files
        sft_path = path / "tool_r1_sft.json"
        if method in {"M0", "M1"} and any((path / name).exists() for name in
                ("tool_r1_sft.json", "tool_r1_checkpoint.json", "tool_r1_export.json")):
            raise ValueError("M0/M1 require an untrained base model, not a Tool-R1 SFT/GRPO export")
        if method == "M2":
            if not sft_path.exists():
                raise ValueError("SFT model needs its tool_r1_sft.json provenance")
            sft = json.loads(sft_path.read_text(encoding="utf-8"))
            verify_merged_files(path, sft)
            result["sft"] = sft
        checkpoint = path / "tool_r1_checkpoint.json"
        if checkpoint.exists():
            result["checkpoint"] = json.loads(checkpoint.read_text(encoding="utf-8"))
            run = Path(result["checkpoint"]["run"]) / "run.json"
            run_info = json.loads(run.read_text(encoding="utf-8"))
            if run_info["args"]["mode"] != method:
                raise ValueError("Evaluation method differs from the exported checkpoint's training mode")
            exported = json.loads((path / "tool_r1_export.json").read_text(encoding="utf-8"))
            verify_merged_files(path, exported)
            result["sft"] = run_info["sft"]
            result["export"] = exported
            result["training_run_sha256"] = digest(run)
            result["training_seed"] = run_info["args"]["seed"]
            result["capacity_test"] = run_info["capacity_test"]
            result["training_args"] = run_info["args"]
            result["training_control"] = _training_control(run_info, result["checkpoint"])
            if run_info["dataset_sha256"] != result["sft"].get("dataset_sha256"):
                raise ValueError("GRPO training and SFT dataset fingerprints differ")
        elif method in {"M3", "M4", "M5"}:
            raise ValueError("GRPO model needs tool_r1_checkpoint.json; use training.export_checkpoint")
    elif method not in {"M0", "M1"}:
        raise ValueError("M2-M5 require a local exported model path for provenance validation")
    return result


def evaluate(data, output, *, method, model_source, split="validation", budgets=(2, 4),
             base_url="http://127.0.0.1:8000/v1", model="tool-r1", seed=42,
             limit=None, max_tokens=384, temperature=0.0, backend="restricted", progress=None):
    if method not in {"M0", "M1", "M2", "M3", "M4", "M5"}:
        raise ValueError("Unknown method")
    if split not in {"validation", "test"}:
        raise ValueError("Model evaluation accepts validation or test only")
    budgets = [0] if method == "M0" else list(budgets)
    if not budgets or len(set(budgets)) != len(budgets) or any(type(b) is not int or not 0 <= b <= 6 for b in budgets):
        raise ValueError("Budgets must be unique integers in [0,6]")
    if max_tokens < 1 or not 0 <= temperature <= 2:
        raise ValueError("Invalid decoding limits")
    data = Path(data)
    audit = _audit(data)
    provenance = _model_provenance(model_source, method)
    if provenance.get("sft", {}).get("dataset_sha256", audit["dataset_sha256"]) != audit["dataset_sha256"]:
        raise ValueError("Model and evaluation dataset fingerprints differ")
    samples = load_jsonl(data / split / "samples.jsonl")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    index = BM25(load_jsonl(data / split / "corpus.jsonl"))
    # Validate URL before writing manifest; URLs must not embed secrets.
    from urllib.parse import urlsplit
    parsed = urlsplit(base_url)
    if parsed.query or parsed.fragment:
        raise ValueError("Base URL must not contain query or fragment credentials")
    HTTPPolicy(base_url, model)
    output = _new_output(output)
    manifest = _manifest(audit, kind="language_model", method=method, split=split,
                         budgets=budgets, seed=seed, backend=backend)
    manifest.update(model=provenance, endpoint=base_url, served_alias=model, limit=limit,
                    decoding={"temperature": temperature, "max_tokens": max_tokens}, budget_visible=method != "M5")
    write_json(output / "manifest.json", manifest)
    traces = []
    with (output / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            for budget in budgets:
                policy_seed = int(hashlib.sha256(f"{seed}/{sample['id']}/{budget}".encode()).hexdigest()[:8], 16) % (2**31)
                policy = HTTPPolicy(base_url, model, max_tokens=max_tokens, temperature=temperature, seed=policy_seed)
                trace = run_episode(sample, policy, index, budget=budget, budget_visible=method != "M5", python_backend=backend,
                                    system_prompt=NO_TOOL_SYSTEM if method == "M0" else None)
                trace.update(evaluation_kind="language_model", method=method,
                             template_family=sample["template_family"], evaluation_seed=seed)
                trace["score"] = score_answer(trace["answer"], sample["answer"], trace["metrics"]["admitted_calls"])
                stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                traces.append(trace)
                if trace["terminal_reason"] == "generation_error":
                    manifest.update(status="failed", reason="model_generation_failed", completed_episodes=len(traces))
                    write_json(output / "manifest.json", manifest)
                    raise RuntimeError("Model generation failed; partial trace saved. Check server log; start a fresh evaluation output.")
                if progress and len(traces) % 20 == 0:
                    progress(len(traces), len(samples) * len(budgets))
    return _save_results(output, traces, manifest)


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Only completed evaluations can be reported or compared")
    if manifest.get("traces_sha256") != digest(directory / "traces.jsonl"):
        raise ValueError("Evaluation traces changed after completion")
    traces = load_jsonl(directory / "traces.jsonl")
    if len(traces) != manifest["episodes"]:
        raise ValueError("Evaluation episode count differs from manifest")
    return manifest, traces


def compare_runs(left, right, *, iterations=2000, seed=42, accuracy_margin=0.02):
    a, left_rows = load_run(left)
    b, right_rows = load_run(right)
    if not 0 <= accuracy_margin < 1:
        raise ValueError("Accuracy margin must lie in [0,1)")
    if a["evaluation_kind"] != "language_model" or b["evaluation_kind"] != "language_model":
        raise ValueError("Method comparisons require genuine language-model evaluations")
    for key in ("dataset_sha256", "code_sha256", "protocol_version", "split", "budgets", "evaluation_seed", "python_backend", "decoding", "scoring"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Incompatible evaluation provenance: {key}")
    # M3/M4/M5 must share SFT initialization for a controlled reward ablation.
    if a["method"] in {"M3", "M4", "M5"} and b["method"] in {"M3", "M4", "M5"}:
        for key in ("sft", "training_seed", "capacity_test", "training_control"):
            if key not in a["model"] or key not in b["model"]:
                raise ValueError(f"RL ablation lacks {key}; re-evaluate with complete training provenance")
            if a["model"].get(key) != b["model"].get(key):
                raise ValueError(f"RL ablation differs in {key}")
        if a["model"]["capacity_test"]:
            raise ValueError("Capacity smoke runs cannot support RL benchmark comparisons")
    result = paired_bootstrap(left_rows, right_rows, iterations=iterations, seed=seed)
    accuracy = result["deltas"]["accuracy"]
    calls = result["deltas"]["mean_admitted_calls"]
    result.update(left=str(left), right=str(right), left_method=a["method"], right_method=b["method"],
                  accuracy_margin=accuracy_margin, accuracy_guardrail_passed=accuracy["lower"] >= -accuracy_margin,
                  significant_call_reduction=calls["upper"] < 0)
    result["efficiency_supported"] = result["accuracy_guardrail_passed"] and result["significant_call_reduction"]
    return result


def write_report(path, manifest, summary):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
    scripted = manifest["evaluation_kind"] == "scripted_smoke"
    lines = ["# Tool-R1 运行报告", "", "**教师脚本工程验证，不是语言模型实验。**" if scripted else "**语言模型评测；数据为合成任务。**", "",
             f"方法：{manifest['method']}；划分：{manifest['split']}；预算：{manifest['budgets']}", "",
             f"数据指纹：`{manifest['dataset_sha256']}`", "",
             "| 分组 | 轨迹数 | 答案正确率 | 平均调用 | 正确题平均调用 | 无效请求率 | 执行失败率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    groups = [("全部", summary["overall"]), *summary["per_task"].items(),
              *(("预算 " + k, v) for k, v in summary["per_budget"].items())]
    for name, row in groups:
        lines.append("| " + " | ".join([name, *(fmt(row[k]) for k in ("n", "accuracy", "mean_admitted_calls", "mean_calls_on_correct", "invalid_request_rate", "execution_failure_rate"))]) + " |")
    lines += ["", "N/A 表示没有适用分母或未测量。全量原始轨迹、运行清单和分任务×预算指标见同目录 JSON 文件。",
              "", "该结果不证明真实领域泛化、RL 收益或跨训练种子的稳定性。效率结论需同数据同预算的模型配对评测及正确率约束。", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
