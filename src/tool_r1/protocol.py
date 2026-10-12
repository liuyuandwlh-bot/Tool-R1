"""Strict event parsing. Environment messages never go through this parser."""
from __future__ import annotations
import json
import math
from .rewards import _number


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number must be finite")
    return number


def parse_action(text: str) -> dict:
    if not isinstance(text, str) or len(text) > 20000:
        raise ValueError("Action must be a JSON string of at most 20000 characters")
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant, parse_float=_finite_float)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Expected one strict JSON object; no fences or commentary") from exc
    if not isinstance(value, dict):
        raise ValueError("Action must be an object")
    if value.get("action") == "tool":
        if set(value) != {"action", "name", "arguments"}:
            raise ValueError("Tool action requires exactly action, name, arguments")
        if value["name"] not in ("retrieve", "python") or not isinstance(value["arguments"], dict):
            raise ValueError("Unknown tool or invalid arguments object")
    elif value.get("action") == "final":
        if set(value) != {"action", "answer"} or not isinstance(value["answer"], dict):
            raise ValueError("Final requires exactly action and answer")
        answer = value["answer"]
        if not {"type", "value"} <= set(answer) or not set(answer) <= {"type", "value", "unit"}:
            raise ValueError("Answer requires type/value and optional unit; scoring metadata is forbidden")
        if answer["type"] not in ("number", "text", "set", "json"):
            raise ValueError("Unknown answer type")
        if answer["type"] == "number":
            number = answer["value"]
            if isinstance(number, bool) or not isinstance(number, (int, float, str)):
                raise ValueError("Numeric value must be a finite number or decimal string")
            if isinstance(number, float) and not math.isfinite(number):
                raise ValueError("Numeric value must be finite")
            _number(number)
            if not isinstance(answer.get("unit"), str) or not answer["unit"].strip():
                raise ValueError("Numeric answer requires explicit unit")
        if answer["type"] == "text" and not isinstance(answer["value"], str):
            raise ValueError("Text answer requires string value")
        if answer["type"] == "set" and not isinstance(answer["value"], list):
            raise ValueError("Set answer requires JSON list")
    else:
        raise ValueError("Action must be tool or final")
    return value

