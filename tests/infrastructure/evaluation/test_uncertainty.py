"""Check uncertainty causes without running a scenario pipeline."""

import pytest

from evaluation.assertions.agent.scenario import check_agent_scenario
from evaluation.assertions.recording import AssertionRecorder
from evaluation.models import EvaluationScenario, PipelineObservation


@pytest.mark.parametrize(
    "tool,status,expected",
    [
        ("GET_HISTORY", "NOT_AVAILABLE", "PASS"),
        ("GET_HISTORY", "NOT_FOUND", "PASS"),
        ("SEARCH_POLICY", "NOT_AVAILABLE", "FAIL"),
        ("SEARCH_PROCEDURE", "NOT_FOUND", "FAIL"),
        ("GET_HISTORY", "OK", "FAIL"),
        (None, "NOT_AVAILABLE", "FAIL"),
    ],
)
def test_history_uncertainty_requires_an_unavailable_history_call(tool, status, expected):
    scenario = EvaluationScenario(
        scenario_id="agent_missing_tool_data",
        description="The historical source must be unavailable.",
        claim_source="controlled",
        expected_uncertainties=["history"],
    )
    observation = PipelineObservation(
        agent={
            "recommendation": {"uncertainties": ["La source est temporairement inaccessible."]},
            "trace": [{"planner_action": "GET_HISTORY", "tool_invoked": tool, "observation_status": status}],
        }
    )
    recorder = AssertionRecorder(scenario, observation)

    check_agent_scenario(recorder)

    checks = {check.name: check for check in recorder.checks}
    assert checks["agent:uncertainty"].status == "PASS"
    assert checks["agent:unavailable_data_observed"].status == expected
