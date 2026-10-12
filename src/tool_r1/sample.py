"""Pre-flight variance sampling for GRPO reward gradient.

Runs K trajectories per question against a vLLM-served base model and reports
per-group reward variance. GRPO needs non-zero group variance to produce a
useful advantage; if K trajectories per question all collapse to the same
reward, GRPO has nothing to optimize against.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from .data import audit_dataset, load_jsonl
from .experiments import write_json
from .policies import HTTPPolicy
from .rewards import score_answer
from .runner import run_episode
from .tools.retrieval import BM25


def _sample_trajectory(sample, *, index, budget, base_url, model, temperature, seed, max_tokens, budget_visible):
    policy = HTTPPolicy(base_url=base_url, model=model, max_tokens=max_tokens,
                        temperature=temperature, seed=seed)
    trace = run_episode(sample, policy, index, budget=budget,
                        budget_visible=budget_visible, python_backend="restricted")
    score = score_answer(trace["answer"], sample["answer"], trace["metrics"]["admitted_calls"])
    return {
        "sample_id": sample["id"],
        "template_family": sample["template_family"],
        "budget": budget,
        "seed": seed,
        "answer": trace["answer"],
        "metrics": trace["metrics"],
        "terminal_reason": trace["terminal_reason"],
        "score": score,
        "reward": float(score["reward"]),
        "correct": bool(score["correctness"]),
        "admitted_calls": int(trace["metrics"]["admitted_calls"]),
    }


def _group_stats(records):
    by_question = {}
    for record in records:
        key = (record["sample_id"], record["budget"])
        by_question.setdefault(key, []).append(record)
    groups = []
    for (sample_id, budget), traj in by_question.items():
        rewards = [t["reward"] for t in traj]
        correctness = [t["correct"] for t in traj]
        calls = [t["admitted_calls"] for t in traj]
        groups.append({
            "sample_id": sample_id,
            "budget": budget,
            "n_trajectories": len(traj),
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "reward_range": max(rewards) - min(rewards),
            "accuracy": sum(correctness) / len(correctness),
            "calls_mean": statistics.fmean(calls),
            "calls_std": statistics.pstdev(calls) if len(calls) > 1 else 0.0,
        })
    return groups


def _aggregate(groups, records):
    reward_values = [g["reward_mean"] for g in groups]
    reward_ranges = [g["reward_range"] for g in groups]
    correct = sum(1 for r in records if r["correct"])
    n_groups_with_variance = sum(1 for g in groups if g["reward_range"] > 1e-6)
    return {
        "n_groups": len(groups),
        "n_records": len(records),
        "reward_mean_of_means": statistics.fmean(reward_values) if reward_values else 0.0,
        "reward_std_of_means": statistics.pstdev(reward_values) if len(reward_values) > 1 else 0.0,
        "reward_range_mean": statistics.fmean(reward_ranges) if reward_ranges else 0.0,
        "reward_range_p50": statistics.median(reward_ranges) if reward_ranges else 0.0,
        "reward_range_p90": (sorted(reward_ranges)[int(0.9 * len(reward_ranges))] if len(reward_ranges) >= 2 else 0.0),
        "share_groups_with_variance": n_groups_with_variance / len(groups) if groups else 0.0,
        "overall_accuracy": correct / len(records) if records else 0.0,
        "verdict": _verdict(groups, records),
    }


def _verdict(groups, records):
    if not groups:
        return "no_data"
    variance_share = sum(1 for g in groups if g["reward_range"] > 1e-6) / len(groups)
    accuracy = sum(1 for r in records if r["correct"]) / len(records) if records else 0.0
    if accuracy > 0.95 and variance_share < 0.40:
        return "saturated_grpo_risk"
    if accuracy < 0.05:
        return "very_low_accuracy_grpo_risk"
    if variance_share < 0.20:
        return "low_variance_grpo_risk"
    return "ok"


def sample(data, output, *, base_url, model, budgets, limit, trajectories, temperature,
           split, seed_eval, max_tokens, budget_visible):
    data = Path(data)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data)
    if not audit["ok"]:
        raise ValueError(f"Dataset audit failed: {audit['errors']}")
    samples = load_jsonl(data / split / "samples.jsonl")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    index = BM25(load_jsonl(data / split / "corpus.jsonl"))
    manifest = {
        "kind": "variance_sample",
        "base_url": base_url,
        "model_alias": model,
        "split": split,
        "trajectories_per_question": trajectories,
        "temperature": temperature,
        "budgets": list(budgets),
        "limit": limit,
        "seed": seed_eval,
        "max_tokens": max_tokens,
        "budget_visible": budget_visible,
        "dataset_sha256": audit["dataset_sha256"],
        "n_questions": len(samples),
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    HTTPPolicy(base_url, model)
    records = []
    total = len(samples) * len(budgets) * trajectories
    started = time.monotonic()
    done = 0
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for sample_id, sample_record in enumerate(samples):
            for budget in budgets:
                for trajectory_idx in range(trajectories):
                    seed_value = seed_eval + 1000 * sample_id + 100 * budget + trajectory_idx
                    record = _sample_trajectory(sample_record, index=index, budget=budget,
                                                base_url=base_url, model=model, temperature=temperature,
                                                seed=seed_value, max_tokens=max_tokens,
                                                budget_visible=budget_visible)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    done += 1
                    if done % 10 == 0:
                        elapsed = time.monotonic() - started
                        rate = done / elapsed if elapsed > 0 else 0
                        print(f"Sampled {done}/{total} ({rate:.1f}/s)", flush=True)
    groups = _group_stats(records)
    aggregate = _aggregate(groups, records)
    per_budget = {}
    for budget in budgets:
        budget_records = [r for r in records if r["budget"] == budget]
        budget_groups = [g for g in groups if g["budget"] == budget]
        per_budget[budget] = _aggregate(budget_groups, budget_records)
    summary = {"overall": aggregate, "per_budget": per_budget, "verdict": aggregate["verdict"]}
    write_json(output / "summary.json", summary)
    write_json(output / "groups.json", groups)
    manifest.update(status="complete", trajectories=done, group_count=len(groups))
    write_json(output / "manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="tool-r1")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--no-budget-visible", action="store_true")
    args = parser.parse_args()
    summary = sample(args.data, args.output, base_url=args.base_url, model=args.model,
                     budgets=args.budgets, limit=args.limit, trajectories=args.trajectories,
                     temperature=args.temperature, split=args.split, seed_eval=args.seed,
                     max_tokens=args.max_tokens, budget_visible=not args.no_budget_visible)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()