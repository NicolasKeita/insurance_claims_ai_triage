"""Explicit inference confidence settings, independently of fusion thresholds."""

import math
import os

CONFIDENCE_THRESHOLDS = (
    ("damage_confidence_threshold", "DAMAGE_CONFIDENCE_THRESHOLD"),
    ("part_confidence_threshold", "PART_CONFIDENCE_THRESHOLD"),
)


def validate_confidence_thresholds(config):
    for field, _ in CONFIDENCE_THRESHOLDS:
        value = getattr(config, field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{field} must be finite and between 0 and 1")


def confidence_thresholds(models, overrides):
    """Explicit options override environment values, then versioned model defaults."""
    values = {key: models[key] for key, _ in CONFIDENCE_THRESHOLDS if key in models}
    for key, variable in CONFIDENCE_THRESHOLDS:
        if overrides.get(key) is None and os.environ.get(variable):
            values[key] = float(os.environ[variable])
    return values
