"""Required reads use the ordinary bounded graph and do not consume LLM calls."""

import pytest

from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentServiceConfig, ClaimsAgentService
from tests.agent.safety._constants import CLAIM_ID
from tests.infrastructure.support.agent_fakes import FakeAgentTools, FakeStructuredLlm, make_recommendation


def test_required_reads_fit_the_tool_budget_without_calling_llm_planner():
    planner_llm = FakeStructuredLlm([])
    tools = FakeAgentTools()
    finalizer_llm = FakeStructuredLlm([make_recommendation()])
    service = ClaimsAgentService(
        AgentServiceConfig(
            tools,
            StructuredAgentPlanner(planner_llm, require_investigation=True),
            StructuredAgentFinalizer(finalizer_llm),
            max_iterations=3,
        )
    )
    result = service.review_claim(CLAIM_ID)
    assert [call[1] for call in tools.calls] == ["GET_CLAIM", "GET_INVESTIGATION", "GET_TRIAGE"]
    assert result.iteration_count == result.tool_call_count == 3
    assert not planner_llm.calls and len(finalizer_llm.calls) == 1
    assert "Maximum tool steps reached." in result.recommendation.uncertainties


@pytest.mark.parametrize("budget", [1, 2])
def test_budget_too_small_for_required_reads_is_rejected_before_io(budget):
    planner = StructuredAgentPlanner(FakeStructuredLlm([]), require_investigation=True)
    with pytest.raises(ValueError, match="required"):
        ClaimsAgentService(AgentServiceConfig(FakeAgentTools(), planner, object(), max_iterations=budget))


@pytest.mark.parametrize(
    "require_investigation,expected_actions",
    [
        (True, ["GET_CLAIM", "GET_INVESTIGATION", "GET_TRIAGE", "GET_HISTORY"]),
        (False, ["GET_CLAIM", "GET_HISTORY"]),
    ],
)
def test_required_history_read_fits_its_declared_budget(require_investigation, expected_actions):
    planner_llm = FakeStructuredLlm([])
    tools = FakeAgentTools()
    finalizer_llm = FakeStructuredLlm([make_recommendation()])
    planner = StructuredAgentPlanner(planner_llm, require_investigation=require_investigation, require_history=True)
    budget = len(expected_actions)
    assert planner.minimum_tool_steps == budget
    service = ClaimsAgentService(
        AgentServiceConfig(tools, planner, StructuredAgentFinalizer(finalizer_llm), max_iterations=budget)
    )

    result = service.review_claim(CLAIM_ID)

    assert [call[1] for call in tools.calls] == expected_actions
    assert result.iteration_count == result.tool_call_count == budget
    assert not planner_llm.calls and len(finalizer_llm.calls) == 1


@pytest.mark.parametrize("require_investigation,budget", [(True, 1), (True, 2), (True, 3), (False, 1)])
def test_budget_too_small_for_required_history_is_rejected_before_io(require_investigation, budget):
    planner = StructuredAgentPlanner(
        FakeStructuredLlm([]), require_investigation=require_investigation, require_history=True
    )
    with pytest.raises(ValueError, match="required"):
        ClaimsAgentService(AgentServiceConfig(FakeAgentTools(), planner, object(), max_iterations=budget))
