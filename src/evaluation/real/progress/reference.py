"""Previous recorded timings are context, never a promised remaining duration."""

import json
import math
from pathlib import Path


def previous_timings(output_dir, scenario_id, model=None):
    path = Path(output_dir) / "observations" / f"{scenario_id}.json"
    try:
        observation = json.loads(path.read_text(encoding="utf-8"))
        if observation.get("provenance", {}).get("mode") != "real":
            return {}
        previous_model = observation.get("model_versions", {}).get("agent_llm")
        if model and previous_model != model:
            return {}
        return {
            name: float(item["duration_seconds"])
            for name, item in observation.get("components", {}).items()
            if isinstance(item.get("duration_seconds"), (int, float))
            and math.isfinite(item["duration_seconds"])
            and item["duration_seconds"] > 0
        }
    except (OSError, ValueError, TypeError, AttributeError):
        return {}
