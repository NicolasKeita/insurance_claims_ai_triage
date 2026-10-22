"""Historic durations are optional references, not fabricated progress percentages."""

import json

from evaluation.real.progress.reference import previous_timings


def test_previous_run_reference_requires_real_mode_and_same_llm(tmp_path):
    directory = tmp_path / "observations"
    directory.mkdir()
    path = directory / "unit.json"
    observation = {
        "provenance": {"mode": "real"},
        "model_versions": {"agent_llm": "installed-model"},
        "components": {"total": {"duration_seconds": 62.5}, "bad": {"duration_seconds": -1}},
    }
    path.write_text(json.dumps(observation))
    assert previous_timings(tmp_path, "unit", "installed-model") == {"total": 62.5}
    assert previous_timings(tmp_path, "unit", "different-model") == {}
    observation["provenance"]["mode"] = "deterministic"
    path.write_text(json.dumps(observation))
    assert previous_timings(tmp_path, "unit") == {}
    path.write_text("incomplete JSON")
    assert previous_timings(tmp_path, "unit") == {}


def test_missing_previous_run_does_not_prevent_monitoring(tmp_path):
    assert previous_timings(tmp_path, "unit") == {}
