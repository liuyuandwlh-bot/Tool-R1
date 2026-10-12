"""Tests for the GRPO variance preflight sampler (analysis logic only)."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from tool_r1.sample import _aggregate, _group_stats, _verdict


def _record(sample_id, budget, trajectory_idx, *, reward, correct, admitted_calls):
    return {
        "sample_id": sample_id,
        "template_family": "calc",
        "budget": budget,
        "seed": trajectory_idx,
        "answer": {"type": "number", "value": 1, "unit": "boxes"},
        "admitted_calls": admitted_calls,
        "correct": correct,
        "reward": reward,
        "terminal_reason": "final",
        "score": {"correctness": correct, "reward": reward, "admitted_calls": admitted_calls},
        "metrics": {"admitted_calls": admitted_calls, "invalid_requests": 0, "execution_failures": 0},
    }


def test_group_stats_measures_range_and_variance():
    records = [
        _record("s1", 4, 0, reward=1.0, correct=True, admitted_calls=0),
        _record("s1", 4, 1, reward=0.9, correct=True, admitted_calls=1),
        _record("s1", 4, 2, reward=0.8, correct=True, admitted_calls=2),
        _record("s1", 4, 3, reward=0.7, correct=True, admitted_calls=3),
        _record("s2", 4, 0, reward=1.0, correct=True, admitted_calls=0),
        _record("s2", 4, 1, reward=1.0, correct=True, admitted_calls=0),
        _record("s2", 4, 2, reward=1.0, correct=True, admitted_calls=0),
        _record("s2", 4, 3, reward=1.0, correct=True, admitted_calls=0),
    ]
    groups = _group_stats(records)
    assert len(groups) == 2
    s1 = next(g for g in groups if g["sample_id"] == "s1")
    s2 = next(g for g in groups if g["sample_id"] == "s2")
    assert s1["reward_range"] == pytest.approx(0.3)
    assert s1["reward_std"] > 0
    assert s2["reward_range"] == 0
    assert s2["reward_std"] == 0


def test_aggregate_flags_low_variance_when_groups_collapse():
    # 5 groups, all collapsed to identical reward.
    records = []
    for s in range(5):
        for t in range(4):
            records.append(_record(f"s{s}", 4, t, reward=1.0, correct=True, admitted_calls=1))
    groups = _group_stats(records)
    aggregate = _aggregate(groups, records)
    assert aggregate["share_groups_with_variance"] == 0
    # High accuracy + zero variance → saturated, not low_variance.
    assert aggregate["verdict"] == "saturated_grpo_risk"


def test_aggregate_flags_saturated_when_high_accuracy_low_variance():
    # All 8 groups correct with zero variance → saturated.
    records = []
    for s in range(8):
        for t in range(4):
            records.append(_record(f"s{s}", 4, t, reward=1.0, correct=True, admitted_calls=0))
    groups = _group_stats(records)
    aggregate = _aggregate(groups, records)
    assert aggregate["overall_accuracy"] == 1.0
    assert aggregate["verdict"] == "saturated_grpo_risk"


def test_aggregate_flags_low_variance_when_groups_collapse_at_partial_accuracy():
    # All 5 groups collapse to the same wrong answer → very low accuracy.
    records = []
    for s in range(5):
        for t in range(4):
            records.append(_record(f"s{s}", 4, t, reward=0.0, correct=False, admitted_calls=1))
    groups = _group_stats(records)
    aggregate = _aggregate(groups, records)
    assert aggregate["share_groups_with_variance"] == 0
    assert aggregate["verdict"] == "very_low_accuracy_grpo_risk"


def test_aggregate_returns_ok_when_variance_and_diversity_exist():
    records = []
    for s in range(8):
        # Half correct, half wrong per question → non-zero reward range.
        for t in range(4):
            correct = t < 2
            reward = 1.0 if correct else 0.0
            records.append(_record(f"s{s}", 4, t, reward=reward, correct=correct, admitted_calls=2))
    groups = _group_stats(records)
    aggregate = _aggregate(groups, records)
    assert aggregate["verdict"] == "ok"
    assert aggregate["overall_accuracy"] == pytest.approx(0.5)


def test_per_budget_breakdown_separates_groups_by_budget():
    records = []
    for s in range(4):
        for budget in (2, 4):
            for t in range(4):
                records.append(_record(f"s{s}", budget, t, reward=0.8, correct=True, admitted_calls=1))
    groups = _group_stats(records)
    by_budget = {budget: [r for r in records if r["budget"] == budget] for budget in (2, 4)}
    per_budget = {budget: _aggregate([g for g in groups if g["budget"] == budget], records_for_budget)
                  for budget, records_for_budget in by_budget.items()}
    assert set(per_budget) == {2, 4}
    for budget, agg in per_budget.items():
        assert agg["n_groups"] == 4