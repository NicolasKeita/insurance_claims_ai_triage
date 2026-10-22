"""Historic durations are optional references, not fabricated progress percentages."""

import json

import pytest

from evaluation.real.progress.reference import previous_timings


def test_previous_run_reference_requires_real_mode_and_same_llm(tmp_path):
    directory = tmp_path / "observations"
    directory.mkdir()
    path = directory / "unit.json"
    observation = {
        "provenance": {"mode": "real"},
        "model_versions": {"agent_llm": "installed-model"},
        "components": {
            "total": {"status": "PASS", "duration_seconds": 62.5},
            "bad": {"status": "PASS", "duration_seconds": -1},
        },
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


@pytest.mark.parametrize(
    "total",
    [
        None,
        {},
        {"duration_seconds": 62.5},
        {"status": "SKIP", "duration_seconds": 62.5},
        *({"status": "PASS", "duration_seconds": value} for value in (0, -1, float("nan"), float("inf"), True)),
    ],
)
def test_previous_run_requires_completed_total_with_usable_duration(tmp_path, total):
    components = {"agent": {"status": "PASS", "duration_seconds": 42}}
    if total is not None:
        components["total"] = total
    directory = tmp_path / "observations"
    directory.mkdir()
    (directory / "unit.json").write_text(json.dumps({"provenance": {"mode": "real"}, "components": components}))
    assert previous_timings(tmp_path, "unit") == {}


@pytest.mark.parametrize("status", ["FAIL", "RUNNING"])
def test_failed_or_interrupted_previous_component_is_not_an_estimate(tmp_path, status):
    directory = tmp_path / "observations"
    directory.mkdir()
    observation = {
        "provenance": {"mode": "real"},
        "components": {
            "total": {"status": "PASS", "duration_seconds": 62.5},
            "agent": {"status": status, "duration_seconds": 42},
        },
    }
    (directory / "unit.json").write_text(json.dumps(observation))
    assert previous_timings(tmp_path, "unit") == {}


def test_reference_keeps_only_positive_finite_successful_durations(tmp_path):
    directory = tmp_path / "observations"
    directory.mkdir()
    components = {
        "total": {"status": "PASS", "duration_seconds": 62.5},
        "agent": {"status": "PASS", "duration_seconds": 42},
        "skipped": {"status": "SKIP", "duration_seconds": 12},
        "unknown": {"duration_seconds": 8},
    }
    for index, value in enumerate((True, False, 0, -1, float("nan"), float("inf"), "2", None, 10**400)):
        components[f"invalid_{index}"] = {"status": "PASS", "duration_seconds": value}
    (directory / "unit.json").write_text(json.dumps({"provenance": {"mode": "real"}, "components": components}))
    assert previous_timings(tmp_path, "unit") == {"total": 62.5, "agent": 42.0}


@pytest.mark.parametrize(
    "observation", [None, [], {"components": []}, {"provenance": {"mode": "real"}, "components": None}]
)
def test_malformed_previous_observation_does_not_prevent_monitoring(tmp_path, observation):
    directory = tmp_path / "observations"
    directory.mkdir()
    (directory / "unit.json").write_text(json.dumps(observation))
    assert previous_timings(tmp_path, "unit") == {}
