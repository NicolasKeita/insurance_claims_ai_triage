"""Denominators and regression gates remain meaningful for partial runs."""

import json

import pytest
from pipeline_support import baseline_report, result

from evaluation.metrics import aggregate_metrics
from evaluation.models import ComponentResult, EndToEndEvaluationReport, EvaluationAssertion
from evaluation.reporting import compare_baseline, create_baseline, write_report


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
