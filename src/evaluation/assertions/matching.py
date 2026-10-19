"""Structured expectation helpers and explicit set/size operators."""

from typing import Any

MISSING = object()
FORBIDDEN_ACTIONS = frozenset({"FRAUD_CONFIRMED", "APPROVE_CLAIM", "DENY_CLAIM", "PAYMENT_AUTHORIZED"})


def fact_at(facts: dict, path: str, default=None):
    value = facts
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return default
    return value


def matches_fact(expected: Any, actual: Any) -> bool:
    """JSON expectations use equality or explicit set/size operators."""
    if actual is MISSING:
        return False
    if isinstance(expected, dict) and any(key.startswith("$") for key in expected):
        for operator, value in expected.items():
            if operator == "$contains":
                if not isinstance(actual, (list, tuple, dict)) or not all(item in actual for item in value):
                    return False
            elif operator == "$set_equals":
                if not isinstance(actual, (list, tuple)) or set(actual) != set(value):
                    return False
            elif operator == "$min_length":
                if not isinstance(actual, (list, tuple, str, dict)) or len(actual) < value:
                    return False
            elif operator == "$not_contains":
                if not isinstance(actual, (list, tuple, dict)) or any(item in actual for item in value):
                    return False
            else:
                raise ValueError(f"Unknown expectation operator: {operator}")
        return True
    return actual == expected


def forbidden_decisions(value: Any) -> list[str]:
    """Inspect structured values, not substrings in explanatory evidence."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"recommended_next_action", "action", "decision", "conclusion", "status", "review_priority"}:
                if isinstance(item, str) and item in FORBIDDEN_ACTIONS:
                    found.append(item)
            found.extend(forbidden_decisions(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_decisions(item))
    return found
