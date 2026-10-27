"""Exact business-field comparison; no fuzzy amount, date or line matching."""

import json
from collections import Counter

CRITICAL_FIELDS = {"claim_id", "incident_date", "total.amount", "total.currency", "declared_damage", "reported_damage"}
SET_FIELDS = {"declared_damage", "reported_damage"}


def _fields(model):
    if model is None:
        return {}
    values = model.model_dump(exclude={"source"})
    flattened = {}
    for name, value in values.items():
        if isinstance(value, dict):
            flattened.update({f"{name}.{key}": item for key, item in value.items()})
        else:
            flattened[name] = value
    return flattened


def _items(values):
    return Counter((item["description"], item["price"]["amount"], item["price"]["currency"]) for item in values)


def _comparable(name, value):
    if value is None:
        return None
    if name in SET_FIELDS:
        return set(value)
    if name == "parts":
        return _items(value)
    return value


def _json_value(name, value):
    if name in SET_FIELDS and value is not None:
        value = sorted(set(value))
    return json.loads(json.dumps(value, default=str))


def _unexpected(name, expected, actual):
    if actual is None or actual == "":
        return 0
    if name in SET_FIELDS:
        return len(set(actual) - set(expected))
    if name == "parts":
        return sum((_items(actual) - _items(expected)).values())
    return int(expected != actual)


def _comparison(name, expected, actual):
    present = actual is not None and actual != ""
    return {
        "field": name,
        "expected": _json_value(name, expected),
        "actual": _json_value(name, actual),
        "matched": present and _comparable(name, expected) == _comparable(name, actual),
        "required": True,
        "present": present,
        "critical": name in CRITICAL_FIELDS,
        "unexpected_value_count": _unexpected(name, expected, actual),
    }


def line_item_metrics(expected, actual) -> dict:
    """Preserve duplicate multiplicity; match description, numeric price and currency."""
    wanted, observed = _items(expected), _items(actual)
    return {
        "expected_count": sum(wanted.values()),
        "extracted_count": sum(observed.values()),
        "matching_items": sum((wanted & observed).values()),
        "missing_items": sum((wanted - observed).values()),
        "unexpected_items": sum((observed - wanted).values()),
    }


def compare_fields(expected, actual) -> dict:
    """All current product model fields are required; provenance is excluded.

    Required recall counts correct values, while required presence also counts
    incorrect observed values. Parse success means a validated model exists.
    """
    reference, observed = _fields(expected), _fields(actual)
    fields = [_comparison(name, value, observed.get(name)) for name, value in reference.items()]
    critical = [field for field in fields if field["critical"]]
    return {
        "fields": fields,
        "exact_match_count": sum(field["matched"] for field in fields),
        "field_count": len(fields),
        "required_field_count": len(fields),
        "required_present_count": sum(field["present"] for field in fields),
        "required_match_count": sum(field["matched"] for field in fields),
        "unexpected_value_count": sum(field["unexpected_value_count"] for field in fields),
        "critical_match_count": sum(field["matched"] for field in critical),
        "critical_field_count": len(critical),
        "parse_success": actual is not None and type(actual) is type(expected),
        "line_items": line_item_metrics(reference["parts"], observed.get("parts", []))
        if "parts" in reference
        else None,
    }
