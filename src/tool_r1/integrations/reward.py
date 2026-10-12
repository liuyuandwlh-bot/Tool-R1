"""Reward from trusted environment results, never text scraped from observations."""

from __future__ import annotations

import json
from tool_r1.rewards import score_answer


def score_episode(answer, ground_truth, metrics, *, cost_weight=0.1, cost_reference=6, truncated=False):
    target = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    result = score_answer(None if truncated else answer, target, int(metrics["admitted_calls"]),
                          cost_weight=cost_weight, cost_reference=cost_reference)
    return {"score": result["reward"], "correctness": float(result["correctness"]), "cost": result["cost"],
            "admitted_calls": result["admitted_calls"], "truncated": float(truncated),
            **{key: int(metrics.get(key, 0)) for key in ("requests", "executed_calls", "invalid_requests", "execution_failures", "budget_violations")}}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Legacy verl fallback. AgentLoop supplies reward_score directly.

Refuse fallback to regex-parsing a mixed assistant/tool transcript: tool output
may contain apparent answers, and text cannot provide trusted call accounting.
"""
    raise RuntimeError("Tool-R1 reward must be computed by ToolR1AgentLoop from its Environment; check the agent config")
