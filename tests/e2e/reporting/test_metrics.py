"""Test metrics."""

import pytest

from evaluation.metrics import aggregate_metrics
from evaluation.models import (
    ComponentResult,
)
from tests.e2e.reporting._support import result


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
