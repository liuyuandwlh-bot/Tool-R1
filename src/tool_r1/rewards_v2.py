"""Improved reward functions for GRPO training with stronger cost optimization signal.

V2 rewards emphasize cost reduction for correct answers, creating stronger gradient signals.
Use these with harder task datasets to enable meaningful RLHF/GRPO training beyond SFT.
"""

from __future__ import annotations

import math
from typing import Any

from .rewards import verify_answer


def score_answer_aggressive(
    prediction: Any,
    target: Any,
    admitted_calls: int,
    cost_weight: float = 0.6,
    cost_reference: int = 6,
) -> dict:
    """Aggressive cost-weighted reward for correct answers in [0.2, 1.0].

    For correct predictions: reward = 1.0 - cost_weight * (cost)^1.5
    - With default cost_weight=0.6, reward range is [0.2, 1.0] (instead of [0.9, 1.0])
    - Quadratic penalty on cost amplifies differences
    - cost^1.5 makes marginal tool use more expensive as cost grows

    For incorrect: reward = 0

    Parameters:
        prediction: Structured answer to verify
        target: Ground truth structured answer
        admitted_calls: Number of tool calls actually made
        cost_weight: Scaling factor for cost penalty (default 0.6 for hard difficulty)
        cost_reference: Budget reference for cost normalization (usually 6)

    Returns:
        Dict with correctness, reward, cost metrics
    """
    if isinstance(admitted_calls, bool) or not isinstance(admitted_calls, int) or admitted_calls < 0:
        raise ValueError("admitted_calls must be a nonnegative integer")
    if isinstance(cost_reference, bool) or not isinstance(cost_reference, int) or cost_reference <= 0:
        raise ValueError("cost_reference must be a positive integer")
    if isinstance(cost_weight, bool) or not isinstance(cost_weight, (float, int)):
        raise ValueError("cost_weight must be finite and in [0, 1]")
    if not math.isfinite(cost_weight) or not 0 <= cost_weight <= 1:
        raise ValueError("cost_weight must be finite and in [0, 1]")

    correctness = verify_answer(prediction, target)
    cost = min(admitted_calls / cost_reference, 1.0)

    # Aggressive penalty: cost^1.5 increases penalty at high cost
    if correctness:
        cost_penalty = cost_weight * (cost ** 1.5)
        reward = max(0.0, 1.0 - cost_penalty)  # Floor at 0 for very high costs
    else:
        reward = 0.0

    return {
        "correctness": correctness,
        "reward": reward,
        "cost": cost,
        "admitted_calls": admitted_calls,
        "cost_weight": cost_weight,
        "cost_reference": cost_reference,
    }


def score_answer_extreme(
    prediction: Any,
    target: Any,
    admitted_calls: int,
    cost_weight: float = 0.8,
    cost_reference: int = 6,
) -> dict:
    """Extreme cost-weighted reward: reward in [0, 1], linear in cost.

    For correct predictions: reward = max(0, 1.0 - cost_weight * cost)
    - With default cost_weight=0.8, reward range is [0.2, 1.0]
    - Linear penalty makes every additional call equally costly
    - Designed for hard tasks with 3-5 required calls

    For incorrect: reward = 0

    This is more aggressive than even `score_answer_aggressive` and suitable for
    scenarios where response quality is already saturated and only cost reduction matters.
    """
    if isinstance(admitted_calls, bool) or not isinstance(admitted_calls, int) or admitted_calls < 0:
        raise ValueError("admitted_calls must be a nonnegative integer")
    if isinstance(cost_reference, bool) or not isinstance(cost_reference, int) or cost_reference <= 0:
        raise ValueError("cost_reference must be a positive integer")
    if isinstance(cost_weight, bool) or not isinstance(cost_weight, (float, int)):
        raise ValueError("cost_weight must be finite and in [0, 1]")
    if not math.isfinite(cost_weight) or not 0 <= cost_weight <= 1:
        raise ValueError("cost_weight must be finite and in [0, 1]")

    correctness = verify_answer(prediction, target)
    cost = min(admitted_calls / cost_reference, 1.0)

    if correctness:
        reward = max(0.0, 1.0 - cost_weight * cost)
    else:
        reward = 0.0

    return {
        "correctness": correctness,
        "reward": reward,
        "cost": cost,
        "admitted_calls": admitted_calls,
        "cost_weight": cost_weight,
        "cost_reference": cost_reference,
    }


def score_answer_cost_only(
    prediction: Any,
    target: Any,
    admitted_calls: int,
    cost_reference: int = 6,
) -> dict:
    """Pure cost optimization reward: 0 if wrong, else 1 - cost.

    For correct predictions: reward = 1.0 - (admitted_calls / cost_reference)
    - Range: [1.0 - cost_max, 1.0] where cost_max = min(calls / reference, 1.0)
    - No weighting, pure linear cost penalty
    - Most extreme: even 1 extra call reduces reward by ~0.167 (1/6)

    For incorrect: reward = 0

    Use when correctness is already 99%+ and you want pure cost gradient.
    """
    if isinstance(admitted_calls, bool) or not isinstance(admitted_calls, int) or admitted_calls < 0:
        raise ValueError("admitted_calls must be a nonnegative integer")
    if isinstance(cost_reference, bool) or not isinstance(cost_reference, int) or cost_reference <= 0:
        raise ValueError("cost_reference must be a positive integer")

    correctness = verify_answer(prediction, target)
    cost = min(admitted_calls / cost_reference, 1.0)

    if correctness:
        reward = 1.0 - cost
    else:
        reward = 0.0

    return {
        "correctness": correctness,
        "reward": reward,
        "cost": cost,
        "admitted_calls": admitted_calls,
        "cost_reference": cost_reference,
    }
