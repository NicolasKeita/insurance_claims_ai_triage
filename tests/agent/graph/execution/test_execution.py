"""Test execution."""

import json

from agent.config import DEFAULT_MAX_ITERATIONS
from agent.models import (
    AgentToolName,
    ObservationStatus,
    RunStatus,
)
from tests.agent.graph._constants import CLAIM_ID
from tests.agent.graph._support import service
from tests.support.agent_fakes import (
    FakeRunStore,
    make_recommendation,
    make_step,
)


def test_explicit_graph_single_tool_finish_and_valid_output():
    store = FakeRunStore()
    agent, planner, finalizer = service(["GET_CLAIM", "FINISH"], store=store)
    result = agent.review_claim(CLAIM_ID)
    assert result.status == RunStatus.COMPLETED
    assert result.iteration_count == result.tool_call_count == 1
    assert result.tools_used == [AgentToolName.GET_CLAIM]
    assert result.recommendation.human_review_required is True
    assert result.recommendation.uncertainties == ["Repair details require human confirmation."]
    assert [step.planner_action.value for step in result.trace] == ["GET_CLAIM", "FINISH"]
    assert result.trace[0].evidence_ids_added == [f"claim:{CLAIM_ID}"]
    assert result.trace[0].tool_args == {}
    assert len(planner.calls) == 2 and len(finalizer.calls) == 1
    assert result.run_id == store.results[0].run_id
    assert result.agent_version == "claims_agent_v2"
    assert result.prompt_version == "claims_agent_prompt_v2"
    assert result.elapsed_seconds >= 0
    json.dumps(result.model_dump(mode="json"), allow_nan=False)
    assert set(agent.graph.get_graph().nodes) == {"__start__", "plan", "tool", "finalize", "validate", "__end__"}


def test_multiple_tools_in_order_evidence_accumulates_finalizer_receives_it():
    steps = ["GET_CLAIM", "GET_ANOMALY", make_step("SEARCH_POLICY", {"query": "hood repair coverage"}), "FINISH"]
    ids = [f"claim:{CLAIM_ID}", f"get_anomaly:{CLAIM_ID}", f"search_policy:{CLAIM_ID}"]
    agent, _, finalizer = service(steps, make_recommendation(ids))
    result = agent.review_claim(CLAIM_ID)
    assert [call[1] for call in agent.tools.calls] == ["GET_CLAIM", "GET_ANOMALY", "SEARCH_POLICY"]
    assert result.iteration_count == 3
    assert [e.evidence_id for e in result.evidence] == ids
    assert all(identifier in finalizer.calls[0][0] for identifier in ids)


def test_maximum_tool_steps_finalizes_instead_of_recursion_crash():
    actions = [
        "GET_CLAIM",
        "GET_HISTORY",
        "GET_TRIAGE",
        "GET_ANOMALY",
        "GET_INVESTIGATION",
        "FIND_SIMILAR_CLAIMS",
        make_step("SEARCH_POLICY", {"query": "collision coverage"}),
        make_step("SEARCH_PROCEDURE", {"query": "review process"}),
    ]
    agent, planner, _ = service(actions)
    result = agent.review_claim(CLAIM_ID)
    assert result.iteration_count == result.tool_call_count == DEFAULT_MAX_ITERATIONS
    assert len(planner.calls) == DEFAULT_MAX_ITERATIONS
    assert "Maximum tool steps reached." in result.recommendation.uncertainties


def test_duplicate_call_is_blocked_deterministically_and_replanning_terminates():
    agent, _, _ = service(["GET_CLAIM", "GET_CLAIM", "GET_CLAIM", "FINISH"])
    result = agent.review_claim(CLAIM_ID)
    assert len(agent.tools.calls) == result.tool_call_count == 1
    assert result.iteration_count == 3
    assert [t.observation_status for t in result.trace[:-1]] == [
        ObservationStatus.OK,
        ObservationStatus.DUPLICATE_BLOCKED,
        ObservationStatus.DUPLICATE_BLOCKED,
    ]
    assert result.trace[1].tool_invoked is None
    assert len(result.evidence) == 1
