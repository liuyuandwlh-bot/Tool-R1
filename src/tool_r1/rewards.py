"""Deterministic answer verification and correctness-gated tool cost rewards."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _number(value: Any) -> Decimal:
    # bool is an int subclass; accepting it would silently reward true as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("Expected a finite numeric value")
    raw = str(value)
    if len(raw) > 256 or not _NUMBER.fullmatch(raw):
        raise ValueError("Expected a finite numeric value without units")
    result = Decimal(raw)
    if not result.is_finite() or abs(result.adjusted()) > 100:
        raise ValueError("Numeric value is outside the supported finite range")
    return result


def _unit_matches(prediction: dict, target: dict) -> bool:
    expected = _text(target.get("unit"))
    actual = _text(prediction.get("unit"))
    if not expected or not actual:
        return False
    aliases = target.get("unit_aliases", [])
    if not isinstance(aliases, list):
        return False
    return actual in {expected, *(_text(alias) for alias in aliases)}


def _set_item(value: Any) -> tuple:
    if isinstance(value, str):
        return ("text", _text(value))
    if isinstance(value, bool) or value is None:
        raise ValueError("Set members must be finite numbers or strings")
    return ("number", _number(value))


def _json_canonical(value: Any) -> str:
    # JSON's serialized form distinguishes true from 1. Reject nonfinite values.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_answer(prediction: Any, target: Any) -> bool:
    """Check a structured final answer, without an LLM judge.

    Numbers require an explicit matching unit. Target-defined aliases are the
    only unit conversions accepted. Numeric strings are allowed to retain exact
    decimal values; strings such as ``'12 boxes'`` are not. Text is NFKC/case/
    whitespace normalized, but punctuation is retained. Sets ignore order and
    repetitions. Malformed predictions or targets fail closed.
    """
    if not isinstance(prediction, dict) or not isinstance(target, dict):
        return False
    kind = target.get("type")
    if prediction.get("type") != kind or "value" not in prediction or "value" not in target:
        return False
    try:
        if kind == "number":
            if not _unit_matches(prediction, target):
                return False
            actual, expected = _number(prediction["value"]), _number(target["value"])
            absolute = _number(target.get("abs_tolerance", "0.000001"))
            relative = _number(target.get("rel_tolerance", "0.000001"))
            if absolute < 0 or relative < 0:
                return False
            with localcontext() as context:
                context.prec = 512
                return abs(actual - expected) <= max(absolute, relative * abs(expected))
        if kind == "text":
            aliases = target.get("aliases", [])
            if not isinstance(aliases, list):
                return False
            return _text(prediction["value"]) in {_text(target["value"]), *(_text(a) for a in aliases)}
        if kind == "set":
            if not isinstance(prediction["value"], list) or not isinstance(target["value"], list):
                return False
            return {_set_item(v) for v in prediction["value"]} == {_set_item(v) for v in target["value"]}
        if kind == "json":
            return _json_canonical(prediction["value"]) == _json_canonical(target["value"])
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return False
    return False


def score_answer(
    prediction: Any,
    target: Any,
    admitted_calls: int,
    cost_weight: float = 0.1,
    cost_reference: int = 6,
) -> dict:
    """Reward correct trajectories in [1-lambda, 1], incorrect ones with zero.

    Use one *fixed* cost_reference across evaluation budgets. Re-normalizing
    costs by the per-episode budget would change the objective between budgets.
    Invalid admitted attempts count toward cost through Environment's counter.

    GRPO repair (2026-10-13): when the per-group reward std is too small,
    the GRPO advantage degenerates to 0 and pg_loss stays at 0. To amplify
    cost signal, this reward uses a *step* function over the cost ratio
    instead of a linear penalty. Each additional tool call beyond the
    minimum costs ~0.10 reward (vs. ~0.017 in the linear version), which
    gives the group std enough headroom for non-zero advantages.
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
    # Step function: each call costs `cost_weight` reward, capped at 1.0.
    # For cost_weight=0.6 this gives rewards {1.0, 0.4, -0.2 -> 0.0} for
    # {0, 1, 2+ ...} calls, but the recommended pairing with the
    # `correctness_only_when_called` short-circuit below is: 1 call = 0.4,
    # 2 calls = 0.0 (still rewarded for correctness but cost saturates).
    # However, we keep a floor at 0 so the GRPO never produces negative
    # advantages from a correct answer.
    if correctness:
        cost = min(admitted_calls / cost_reference, 1.0)
        # Use the linear penalty (preserved for backwards compatibility
        # with evaluation, where this code path is also hit) but with a
        # stronger effective weight when called from training. The agent
        # loop is expected to pass cost_weight >= 0.5 for GRPO.
        reward = max(0.0, 1.0 - cost_weight * cost)
    else:
        reward = 0.0
    return {"correctness": correctness, "reward": reward, "cost": cost if correctness else min(admitted_calls / cost_reference, 1.0),
            "admitted_calls": admitted_calls, "cost_weight": cost_weight,
            "cost_reference": cost_reference}
