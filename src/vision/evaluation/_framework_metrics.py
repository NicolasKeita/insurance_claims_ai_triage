"""Honest parsing of framework metric groups into per-class records."""

import math
from typing import Any

METRIC_FIELDS = ("precision", "recall", "map50", "map50_95")
_OVERALL_ATTRIBUTES = ("mp", "mr", "map50", "map")


def metric_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if not 0 <= number <= 1:
        raise ValueError(f"Metric outside [0,1]: {number}")
    return number


def overall_metrics(group: Any) -> dict[str, float | None]:
    return {key: metric_number(getattr(group, attr, None)) for key, attr in zip(METRIC_FIELDS, _OVERALL_ATTRIBUTES)}


def class_metrics(group: Any) -> dict[int, dict[str, float | None]]:
    if group is None:
        return {}
    indices = getattr(group, "ap_class_index", [])
    if indices is None:
        return {}
    class_result = getattr(group, "class_result", None)
    if not callable(class_result):
        return {}
    values = {}
    for position, class_id in enumerate(indices):
        row = class_result(position)
        values[int(class_id)] = {
            key: metric_number(row[index]) if len(row) > index else None for index, key in enumerate(METRIC_FIELDS)
        }
    return values