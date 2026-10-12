"""Trace summaries and paired, stratified bootstrap without ML dependencies."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from statistics import mean
from typing import Iterable


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _row(trace: dict) -> dict:
    if not isinstance(trace, dict):
        raise ValueError("Traces must be dictionaries")
    score = trace.get("score", {})
    correct = trace.get("correctness", score.get("correctness"))
    if not isinstance(correct, bool):
        raise ValueError("Each trace must have a scored boolean correctness field")
    sample_id, task, budget = trace.get("sample_id"), trace.get("task_type"), trace.get("budget")
    if not isinstance(sample_id, str) or not sample_id or not isinstance(task, str) or not task:
        raise ValueError("Each trace requires sample_id and task_type strings")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Each trace requires a nonnegative integer budget")
    metrics = trace.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    counts = {}
    for name in ("requests", "admitted_calls", "executed_calls", "invalid_requests", "execution_failures", "budget_violations"):
        value = metrics.get(name, score.get(name, trace.get(name, 0)))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        counts[name] = value
    reward = _finite(trace.get("reward", score.get("reward", float(correct))), "reward")
    cost = _finite(trace.get("cost", score.get("cost", 0.0)), "cost")
    if not 0 <= cost <= 1:
        raise ValueError("cost must be in [0, 1]")
    tokens = trace.get("generated_tokens", metrics.get("generated_tokens"))
    if metrics.get("generation_token_count_available") is False:
        tokens = None
    if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
        raise ValueError("generated_tokens must be a nonnegative integer when supplied")
    return {"sample_id": sample_id, "task_type": task, "budget": budget,
            "correctness": correct, "reward": reward, "cost": cost,
            "generated_tokens": tokens, "terminal_reason": trace.get("terminal_reason", "unspecified"),
            "evaluation_kind": trace.get("evaluation_kind", "unspecified"), **counts}


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    successful = [row for row in rows if row["correctness"]]
    direct_rows = [row for row in rows if row["task_type"] == "direct"]
    sums = {name: sum(row[name] for row in rows) for name in
            ("requests", "admitted_calls", "executed_calls", "invalid_requests", "execution_failures", "budget_violations")}
    token_rows = [row["generated_tokens"] for row in rows if row["generated_tokens"] is not None]
    def average(name):
        return mean(row[name] for row in rows) if n else None
    return {
        "n": n, "correct": len(successful), "accuracy": len(successful) / n if n else None,
        "mean_reward": average("reward"), "mean_cost": average("cost"),
        "mean_admitted_calls": average("admitted_calls"), "mean_tool_calls": average("admitted_calls"),
        "mean_executed_calls": average("executed_calls"),
        "mean_calls_on_correct": mean(row["admitted_calls"] for row in successful) if successful else None,
        "invalid_request_rate": sums["invalid_requests"] / sums["requests"] if sums["requests"] else None,
        "execution_failure_rate": sums["execution_failures"] / sums["executed_calls"] if sums["executed_calls"] else None,
        "direct_episodes": len(direct_rows),
        "direct_unnecessary_tool_episode_rate": sum(row["admitted_calls"] > 0 for row in direct_rows) / len(direct_rows) if direct_rows else None,
        "budget_violation_episode_rate": sum(row["budget_violations"] > 0 for row in rows) / n if n else None,
        "mean_generated_tokens": mean(token_rows) if token_rows else None,
        "generated_token_coverage": len(token_rows), "totals": sums,
        "terminal_reasons": dict(sorted(Counter(row["terminal_reason"] for row in rows).items())),
    }


def summarize(traces: Iterable[dict]) -> dict:
    """Summarize scored traces, keeping each task and budget visible.

    The caller should supply evaluation_kind='language_model' or
    'scripted_smoke'. Unlabelled input remains explicitly unspecified.
    Undefined rates are None. The unnecessary-tool rate uses direct-task
    episodes as its denominator and counts any episode with admitted_calls>0.
    """
    rows = [_row(trace) for trace in traces]
    by_task, by_budget, by_task_budget = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
        by_budget[row["budget"]].append(row)
        by_task_budget[(row["task_type"], row["budget"])].append(row)
    return {
        "overall": _aggregate(rows),
        "per_task": {task: _aggregate(group) for task, group in sorted(by_task.items())},
        "per_budget": {str(budget): _aggregate(group) for budget, group in sorted(by_budget.items())},
        "per_task_budget": {f"{task}/budget={budget}": _aggregate(group) for (task, budget), group in sorted(by_task_budget.items())},
        "evaluation_kinds": dict(sorted(Counter(row["evaluation_kind"] for row in rows).items())),
        "interpretation": "Scripted smoke checks validate software only. Only language_model traces measure a model. Synthetic-data results do not establish real-world task performance.",
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(
    left: Iterable[dict], right: Iterable[dict], iterations: int = 2000,
    seed: int = 42, confidence: float = 0.95,
) -> dict:
    """Compare right minus left on exactly aligned sample/budget pairs.

    Resample sample-ID clusters, preserving correlations across budgets.
    Stratify by task type and observed budget schedule, preserving task/budget
    composition. This is evaluation-sample uncertainty, not training-seed
    uncertainty. It must not be used to claim reproducibility across RL runs.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    confidence = _finite(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    def index(traces, label):
        mapping = {}
        for trace in traces:
            row = _row(trace)
            key = (row["sample_id"], row["budget"])
            if key in mapping:
                raise ValueError(f"Duplicate sample_id/budget pair in {label}: {key}; compare one evaluation seed at a time")
            mapping[key] = row
        return mapping

    left_index, right_index = index(left, "left"), index(right, "right")
    if not left_index or left_index.keys() != right_index.keys():
        raise ValueError("Paired comparison requires identical nonempty sample_id/budget sets")
    clusters, task_by_sample = defaultdict(list), {}
    for key in sorted(left_index):
        a, b = left_index[key], right_index[key]
        if a["task_type"] != b["task_type"]:
            raise ValueError(f"Task mismatch for pair {key}")
        if key[0] in task_by_sample and task_by_sample[key[0]] != a["task_type"]:
            raise ValueError(f"Task type varies across budgets for sample {key[0]}")
        task_by_sample[key[0]] = a["task_type"]
        clusters[key[0]].append({"budget": key[1],
                                "accuracy": float(b["correctness"]) - float(a["correctness"]),
                                "mean_admitted_calls": b["admitted_calls"] - a["admitted_calls"],
                                "mean_reward": b["reward"] - a["reward"]})
    strata = defaultdict(list)
    for sample_id, pairs in sorted(clusters.items()):
        strata[(task_by_sample[sample_id], tuple(pair["budget"] for pair in pairs))].append(pairs)
    metric_names = ("accuracy", "mean_admitted_calls", "mean_reward")
    observed = {name: sum(pair[name] for pairs in clusters.values() for pair in pairs) / len(left_index) for name in metric_names}
    samples = {name: [] for name in metric_names}
    rng = random.Random(seed)
    # Cluster metric sums avoid rescanning each budget row in every replicate.
    summarized_strata = []
    for group in strata.values():
        summarized_strata.append([tuple(sum(pair[name] for pair in cluster) for name in metric_names) for cluster in group])
    for _ in range(iterations):
        totals = [0.0] * len(metric_names)
        for group in summarized_strata:
            for _ in range(len(group)):
                selected = group[rng.randrange(len(group))]
                for index_, value in enumerate(selected):
                    totals[index_] += value
        for name, total in zip(metric_names, totals):
            samples[name].append(total / len(left_index))
    tail = (1 - confidence) / 2
    result = {
        "direction": "right_minus_left", "n_pairs": len(left_index), "n_samples": len(clusters),
        "iterations": iterations, "seed": seed, "confidence": confidence,
        "method": "paired sample-cluster percentile bootstrap, stratified by task and observed budget schedule",
        "strata": [{"task_type": task, "budgets": list(budgets), "samples": len(group)}
                   for (task, budgets), group in sorted(strata.items())],
        "deltas": {name: {"estimate": observed[name], "lower": _quantile(samples[name], tail),
                          "upper": _quantile(samples[name], 1 - tail)} for name in metric_names},
        "limitations": "Confidence intervals quantify evaluation-sample uncertainty only; they do not include training-seed variation. Single-sample strata cannot estimate within-stratum variation.",
    }
    return result


compare_paired = paired_bootstrap
