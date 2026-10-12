"""Export audited data to verl parquet and replay training demonstrations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tool_r1.data import audit_dataset, load_jsonl
from tool_r1.rewards import verify_answer


def build_rl_rows(samples: list[dict], *, budgets=(2, 4), budget_visible=True) -> list[dict]:
    from tool_r1.runner import Environment
    rows = []
    for sample in samples:
        if sample["split"] not in {"train", "validation"}:
            raise ValueError("Training export accepts train/validation only; test remains sealed")
        for budget in budgets:
            env = Environment([], budget=budget, budget_visible=budget_visible)
            rows.append({
                "data_source": "tool_r1", "agent_name": "tool_r1",
                "prompt": env.initial_messages(sample["question"]),
                "ability": "tool_reasoning",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(sample["answer"], ensure_ascii=False)},
                "extra_info": {"index": len(rows), "sample_id": sample["id"], "split": sample["split"],
                               "question": sample["question"], "budget": int(budget),
                               "budget_visible": budget_visible, "task_type": sample["task_type"]},
            })
    return rows


def replay_demonstrations(data_dir: str | Path, *, budgets=(2, 4), backend="restricted") -> list[dict]:
    """Execute every teacher action, validate observations and final answer.

Teacher targets are supervised training data. Only question + environment
observations enter each input; no sample metadata or answer key is supplied.
"""
    from tool_r1.runner import Environment
    from tool_r1.tools.retrieval import BM25
    root = Path(data_dir)
    samples = {s["id"]: s for s in load_jsonl(root / "train/samples.jsonl")}
    docs = BM25(load_jsonl(root / "train/corpus.jsonl"))
    demonstrations = load_jsonl(root / "train/demonstrations.jsonl")
    seen, examples = set(), []
    for demo in demonstrations:
        sid = demo["sample_id"]
        if demo.get("split") != "train" or sid not in samples or sid in seen:
            raise ValueError(f"Nontraining, unknown or duplicated demo: {sid}")
        seen.add(sid)
        sample = samples[sid]
        # Use enough budget to replay the demonstration; the reward function still
        # penalizes tool use, so the model learns to be efficient at inference time.
        n_tool_actions = sum(1 for a in demo["reference_actions"] if a.get("action") == "tool")
        base_budget = budgets[int(hashlib.sha256(sid.encode()).hexdigest(), 16) % len(budgets)]
        budget = max(base_budget, n_tool_actions)
        env = Environment(docs, budget=budget, budget_visible=True, python_backend=backend)
        messages = env.initial_messages(sample["question"])
        observed = ""
        python_results = []
        final = None
        actions = demo["reference_actions"]
        for index, action in enumerate(actions):
            text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            example = {"sample_id": sid, "split": "train", "messages": copy.deepcopy(messages), "action": text}
            result = env.step(text)
            messages.append({"role": "assistant", "content": text})
            if result.get("observation"):
                observed += result["observation"]
                messages.append({"role": "user", "content": result["observation"]})
                if action.get("action") == "tool" and action.get("name") == "python":
                    observation = json.loads(result["observation"])
                    if observation.get("ok") is not True or not isinstance(observation.get("result"), str):
                        raise ValueError(f"{sid}: Python teacher execution produced no successful stdout")
                    python_results.append(observation["result"].strip())
            examples.append(example)
            if result["done"]:
                if index != len(actions) - 1:
                    raise ValueError(f"{sid}: demonstration continues after termination")
                final = result.get("answer")
        if not verify_answer(final, sample["answer"]):
            raise ValueError(f"{sid}: demonstration failed answer verification")
        if any(env.metrics.get(k, 0) for k in ("invalid_requests", "execution_failures", "budget_violations")):
            raise ValueError(f"{sid}: demonstration tool execution failed: {env.metrics}")
        if not all(doc_id in observed for doc_id in sample["source_doc_ids"]):
            raise ValueError(f"{sid}: teacher uses evidence absent from actual retrieval observations")
        if sample["task_type"] in {"compute", "compose"}:
            prediction = {"type": "number", "value": python_results[-1] if python_results else None,
                          "unit": sample["answer"]["unit"]}
            if not verify_answer(prediction, sample["answer"]):
                raise ValueError(f"{sid}: actual Python stdout does not support the teacher's final answer")
    if seen != set(samples):
        raise ValueError("Demonstrations must cover exactly the training sample IDs")
    return examples


def prepare(data_dir: str | Path, output_dir: str | Path) -> dict:
    root, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    audit = audit_dataset(root)
    if not audit["ok"]:
        raise ValueError(f"Data audit failed: {audit['errors']}")
    examples = replay_demonstrations(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sft_turns.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the server training dependencies (pyarrow) to export parquet") from exc
    files = []
    for visible in (True, False):
        for split in ("train", "validation"):
            rows = build_rl_rows(load_jsonl(root / split / "samples.jsonl"), budget_visible=visible)
            if not rows:
                raise ValueError(f"{split} must contain samples")
            name = f"{split}_{'visible' if visible else 'hidden'}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), output / name)
            files.append({"name": name, "rows": len(rows), "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()})
    manifest = {"dataset_sha256": audit["dataset_sha256"], "budgets": [2, 4],
                "sft_turns": len(examples), "demonstrations_replayed": True,
                "files": files, "source": "synthetic deterministic training teacher", "test_exported": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="data/prepared")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
