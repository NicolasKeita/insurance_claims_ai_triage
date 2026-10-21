"""Regression comparison against a frozen real baseline."""

import pytest
from pipeline_support import baseline_report

from evaluation.models import EvaluationAssertion
from evaluation.reporting import baseline_snapshot, compare_baseline


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
def test_regression_comparison_detects_stable_behavior_regression(check_name):
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
