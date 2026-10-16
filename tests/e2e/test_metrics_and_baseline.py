"""Denominators and regression gates remain meaningful for partial runs."""

import json

import pytest

from evaluation.metrics import aggregate_metrics
from evaluation.models import (
    ComponentResult,
    EndToEndEvaluationReport,
    EvaluationAssertion,
    PipelineObservation,
    ScenarioEvaluationResult,
)
from evaluation.reporting import baseline_snapshot, compare_baseline, create_baseline, write_report


def result(scenario_id, status, *, checks=None, components=None):
    return ScenarioEvaluationResult(
        scenario_id=scenario_id,
        status=status,
        assertions=checks or [],
        component_results=components or {},
        observation=PipelineObservation(
            model_versions={"anomaly": "model-v1"}, policy_versions={"investigation": "policy-v3"}
        ),
    )


def test_skips_never_inflate_pass_rate_and_no_executions_have_no_rate():
    rows = [result("passed", "PASS"), result("failed", "FAIL"), result("unavailable", "SKIP")]
    metrics = aggregate_metrics(rows)
    assert metrics["scenario_pass_rate"] == 0.5
    assert metrics["scenario_pass_rate_denominator"] == 2
    assert metrics["scenario_counts"] == {"PASS": 1, "FAIL": 1, "SKIP": 1}
    assert aggregate_metrics([rows[-1]])["scenario_pass_rate"] is None


def test_component_completion_separates_attempts_from_unavailable_requests():
    rows = [
        result("completed", "PASS", components={"vision": ComponentResult(status="PASS")}),
        result("failed", "FAIL", components={"vision": ComponentResult(status="FAIL")}),
        result("missing_checkpoint", "SKIP", components={"vision": ComponentResult(status="SKIP")}),
    ]
    completion = aggregate_metrics(rows)["component_completion"]["vision"]
    assert completion["success_rate"] == 0.5
    assert completion["requested_completion_rate"] == pytest.approx(1 / 3)
    assert (completion["attempted"], completion["skipped"]) == (2, 1)


def test_agent_completion_counts_a_failure_without_a_result_payload():
    rows = [
        result("completed", "PASS", components={"agent": ComponentResult(status="PASS")}),
        result("llm_exception", "FAIL", components={"agent": ComponentResult(status="FAIL")}),
        result("ollama_unavailable", "SKIP", components={"agent": ComponentResult(status="SKIP")}),
    ]
    rows[0].observation.agent = {"status": "COMPLETED"}
    assert aggregate_metrics(rows)["agent_completion_rate"] == 0.5
    assert aggregate_metrics([rows[-1]])["agent_completion_rate"] is None


def baseline_report():
    checks = [
        EvaluationAssertion(name=name, component=component, status="PASS", expected=True, actual=True)
        for name, component in [
            ("required_signal:ADDITIONAL_QUOTE_DAMAGE", "investigation"),
            ("forbidden_signal:VISUAL_ADDITIONAL_DAMAGE", "investigation"),
            ("rag:expected_sources", "retrieval"),
            ("rag:citation_validity", "retrieval"),
            ("agent:evidence_citations", "agent"),
        ]
    ]
    return EndToEndEvaluationReport(
        mode="real", full_stack_status="COMPLETE", results=[result("golden", "PASS", checks=checks)]
    )


@pytest.mark.parametrize(
    "check_name",
    [
        "required_signal:ADDITIONAL_QUOTE_DAMAGE",
        "forbidden_signal:VISUAL_ADDITIONAL_DAMAGE",
        "rag:expected_sources",
        "rag:citation_validity",
        "agent:evidence_citations",
    ],
)
def test_regression_comparison_detects_stable_behavior_regressions(check_name):
    report = baseline_report()
    baseline = baseline_snapshot(report)
    current = report.model_copy(deep=True)
    check = next(check for check in current.results[0].assertions if check.name == check_name)
    check.status = "FAIL"
    current.results[0].status = "FAIL"
    regressions = compare_baseline(current, baseline)
    assert {item["kind"] for item in regressions} >= {"scenario_regression", check_name}


def test_changing_expectations_under_the_same_version_is_a_regression():
    report = baseline_report()
    report.results[0].assertions.append(
        EvaluationAssertion(
            name="fact:history.claims_last_30_days", component="history", status="PASS", expected=3, actual=3
        )
    )
    baseline = baseline_snapshot(report)
    changed = report.model_copy(deep=True)
    check = changed.results[0].assertions[-1]
    check.expected = check.actual = 2
    regressions = compare_baseline(changed, baseline)
    mismatch = next(item for item in regressions if item["kind"] == "benchmark_expectation_changed")
    assert (mismatch["expectation"], mismatch["expected"], mismatch["actual"]) == (
        "fact:history.claims_last_30_days",
        3,
        2,
    )


def test_partial_real_or_deterministic_runs_cannot_create_real_baseline(tmp_path):
    report = baseline_report()
    path = tmp_path / "baseline_summary.json"
    report.full_stack_status = "PARTIAL"
    assert create_baseline(report, path) is False
    assert not path.exists()
    report.full_stack_status = "COMPLETE"
    report.mode = "deterministic"
    assert create_baseline(report, path) is False
    assert not path.exists()
    report.mode = "real"
    report.results[0].status = "FAIL"
    assert create_baseline(report, path) is False
    assert not path.exists()


def test_baseline_ignores_latency_and_records_versions(tmp_path):
    report = baseline_report()
    report.results[0].timings = {"total": 1.25}
    path = tmp_path / "baseline_summary.json"
    assert create_baseline(report, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["scenarios"]["golden"]["model_versions"] == {"anomaly": "model-v1"}
    assert "timings" not in saved["scenarios"]["golden"]
    report.results[0].timings["total"] = 999
    assert compare_baseline(report, saved) == []
    saved["benchmark_version"] = "e2e_claims_future"
    with pytest.raises(ValueError, match="version mismatch"):
        compare_baseline(report, saved)


def test_failure_artifacts_include_expected_actual_component_and_trace(tmp_path):
    check = EvaluationAssertion(
        name="fact:history.claims_last_30_days",
        component="history",
        status="FAIL",
        expected=3,
        actual=4,
        source="traces/history.json#history",
    )
    report = EndToEndEvaluationReport(mode="deterministic", results=[result("history", "FAIL", checks=[check])])
    output = write_report(report, tmp_path)
    failure = json.loads((output / "failures/history.json").read_text(encoding="utf-8"))
    assertion = failure["assertions"][0]
    assert (assertion["expected"], assertion["actual"], assertion["component"], assertion["source"]) == (
        3,
        4,
        "history",
        "traces/history.json#history",
    )
    assert all(
        (output / name).is_file()
        for name in [
            "summary.json",
            "scenario_results.json",
            "scenario_results.csv",
            "timings.csv",
            "traces/history.json",
        ]
    )
