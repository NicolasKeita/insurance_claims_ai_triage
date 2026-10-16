"""Stable regression snapshots that exclude latency, prose and confidences."""

import json
from pathlib import Path

from evaluation.reporting.artifacts import _json

SNAPSHOT_PREFIXES = (
    "fact:",
    "required_signal:",
    "forbidden_signal:",
    "rag:expected_sources",
    "rag:expected_section",
    "rag:citation_validity",
    "agent:evidence_citations",
    "agent:acceptable_action",
    "agent:human_review_required",
    "agent:final_schema",
    "schema:",
)


def baseline_snapshot(report):
    """No exact latency, prose or floating point confidence is retained."""
    return {
        "benchmark_version": report.benchmark_version,
        "mode": report.mode,
        "full_stack_status": report.full_stack_status,
        "scenarios": {
            result.scenario_id: {
                "status": result.status,
                "structured_expectations": {
                    assertion.name: {"expected": assertion.expected, "status": assertion.status}
                    for assertion in result.assertions
                    if assertion.name.startswith(SNAPSHOT_PREFIXES)
                },
                "signals": result.observation.signals,
                "model_versions": result.observation.model_versions,
                "policy_versions": result.observation.policy_versions,
            }
            for result in report.results
        },
    }


def create_baseline(report, path):
    """Withhold a real-stack baseline until the complete requested stack passed."""
    eligible = (
        report.mode == "real"
        and report.full_stack_status == "COMPLETE"
        and report.results
        and all(result.status == "PASS" for result in report.results)
    )
    if not eligible:
        return False
    _json(Path(path), baseline_snapshot(report))
    return True


def _expectation_regressions(scenario_id, previous, actual):
    regressions = []
    for name, expectation in previous["structured_expectations"].items():
        now = actual["structured_expectations"].get(name)
        if now is not None and now["expected"] != expectation["expected"]:
            regressions.append(
                {
                    "scenario_id": scenario_id,
                    "kind": "benchmark_expectation_changed",
                    "expectation": name,
                    "expected": expectation["expected"],
                    "actual": now["expected"],
                }
            )
        if expectation["status"] == "PASS" and (now is None or now["status"] != "PASS"):
            regressions.append({"scenario_id": scenario_id, "kind": name, "expected": expectation, "actual": now})
    return regressions


def _new_forbidden_signals(scenario_id, previous, actual):
    return [
        {"scenario_id": scenario_id, "kind": name, "expected": "absent", "actual": "present"}
        for name, now in actual["structured_expectations"].items()
        if name.startswith("forbidden_signal:")
        and now["status"] == "FAIL"
        and name not in previous["structured_expectations"]
    ]


def _compare_scenario(scenario_id, previous, actual):
    if actual is None:
        return [
            {"scenario_id": scenario_id, "kind": "missing_scenario", "expected": previous["status"], "actual": None}
        ]
    regressions = []
    if previous["status"] == "PASS" and actual["status"] != "PASS":
        regressions.append(
            {"scenario_id": scenario_id, "kind": "scenario_regression", "expected": "PASS", "actual": actual["status"]}
        )
    regressions.extend(_expectation_regressions(scenario_id, previous, actual))
    regressions.extend(_new_forbidden_signals(scenario_id, previous, actual))
    return regressions


def compare_baseline(report, baseline):
    """Forbidden signals are gates, not all newly observed legitimate ML signals."""
    if isinstance(baseline, (str, Path)):
        baseline = json.loads(Path(baseline).read_text(encoding="utf-8"))
    if baseline["benchmark_version"] != report.benchmark_version:
        raise ValueError("Benchmark version mismatch; scenario expectations cannot be compared")
    if baseline.get("mode") != report.mode:
        raise ValueError("Evaluation mode mismatch; real and deterministic baselines cannot be compared")
    current = baseline_snapshot(report)
    regressions = []
    for scenario_id, previous in baseline["scenarios"].items():
        regressions.extend(_compare_scenario(scenario_id, previous, current["scenarios"].get(scenario_id)))
    return regressions
