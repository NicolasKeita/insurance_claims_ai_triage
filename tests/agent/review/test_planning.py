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
            max_iterations=2,
        )
    )
    result = service.review_claim(CLAIM_ID)
    assert [call[1] for call in tools.calls] == ["GET_CLAIM", "GET_INVESTIGATION"]
    assert result.iteration_count == result.tool_call_count == 2
    assert not planner_llm.calls and len(finalizer_llm.calls) == 1
    assert "Maximum tool steps reached." in result.recommendation.uncertainties


def test_budget_too_small_for_required_reads_is_rejected_before_io():
    planner = StructuredAgentPlanner(FakeStructuredLlm([]), require_investigation=True)
    with pytest.raises(ValueError, match="required"):
        ClaimsAgentService(AgentServiceConfig(FakeAgentTools(), planner, object(), max_iterations=1))
