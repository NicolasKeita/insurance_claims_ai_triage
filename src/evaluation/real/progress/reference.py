"""Load usable timings from a completed, successful real observation."""

import json
import math
from pathlib import Path


def _duration(item):
    if item.get("status") != "PASS":
        return None
    value = item.get("duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        duration = float(value)
    except OverflowError:
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def previous_timings(output_dir, scenario_id, model=None):
    path = Path(output_dir) / "observations" / f"{scenario_id}.json"
    try:
        observation = json.loads(path.read_text(encoding="utf-8"))
        if observation.get("provenance", {}).get("mode") != "real":
            return {}
        previous_model = observation.get("model_versions", {}).get("agent_llm")
        if model and previous_model != model:
            return {}
        components = observation.get("components", {})
        if any(item.get("status") in {"FAIL", "RUNNING"} for item in components.values()):
            return {}
        if _duration(components.get("total", {})) is None:
            return {}
        return {name: duration for name, item in components.items() if (duration := _duration(item)) is not None}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}
