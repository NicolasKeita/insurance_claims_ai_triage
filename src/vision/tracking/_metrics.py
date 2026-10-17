"""Flatten validation and TEST metrics into stable MLflow metric keys."""

from typing import Any


def flatten_metrics(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}.{group}.{key}": value
        for group in ("box", "mask")
        for key, value in payload.get(group, {}).items()
        if value is not None
    }


def per_class_metrics(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    """Keep real absent-class values absent, with stable taxonomy metric keys."""
    metrics = {}
    for row in payload.get("per_class", []):
        name = str(row.get("damage_type") or row["name"].replace(" ", "_"))
        metrics.update(flatten_metrics(row, f"{prefix}.class.{name}"))
    return metrics
