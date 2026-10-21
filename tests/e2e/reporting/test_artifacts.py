"""Test artifacts."""

import json

from evaluation.models import (
    EndToEndEvaluationReport,
    EvaluationAssertion,
)
from evaluation.reporting import (
    write_report,
)
from tests.e2e.reporting._support import result


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
