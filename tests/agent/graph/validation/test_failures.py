"""Test failures."""

import pytest

from agent.models import (
    AgentToolName,
    RunStatus,
)
from agent.service import AgentRunError
from tests.agent.graph._constants import CLAIM_ID
from tests.agent.graph._support import service
from tests.support.agent_fakes import (
    FakeAgentTools,
    FakeRunStore,
    make_recommendation,
)


def test_unexpected_tool_failure_fails_and_persists_sanitized_trace():
    store = FakeRunStore()
    tools = FakeAgentTools(failure=RuntimeError("secret password and raw document"))
    agent, _, finalizer = service(["GET_CLAIM"], tools=tools, store=store)
    with pytest.raises(AgentRunError) as error:
        agent.review_claim(CLAIM_ID)
    assert error.value.failure_kind == "TOOL"
    saved = store.results[0]
    assert saved.status == RunStatus.FAILED and saved.recommendation is None
    assert saved.tool_call_count == 1
    assert saved.trace[0].tool_invoked == AgentToolName.GET_CLAIM
    assert "secret" not in saved.model_dump_json()
    assert not finalizer.calls


def test_invalid_planner_generation_is_controlled_failure():
    store = FakeRunStore()
    agent, _, _ = service([{"action": "SQL", "reason": "bad", "arguments": {}}], store=store)
    with pytest.raises(AgentRunError) as error:
        agent.review_claim(CLAIM_ID)
    assert error.value.failure_kind == "GENERATION"
    assert store.results[0].iteration_count == 0
    assert not agent.tools.calls


def test_unknown_evidence_citation_is_rejected_and_failed_run_saved():
    store = FakeRunStore()
    agent, _, _ = service(["GET_CLAIM", "FINISH"], make_recommendation(["invented:claim"]), store=store)
    with pytest.raises(AgentRunError) as error:
        agent.review_claim(CLAIM_ID)
    assert error.value.failure_kind == "GROUNDING"
    assert store.results[0].status == RunStatus.FAILED
    assert store.results[0].recommendation is None


def test_invalid_finalizer_action_is_rejected():
    output = make_recommendation().model_dump()
    output["recommended_next_action"] = "APPROVE_CLAIM"
    agent, _, _ = service(["GET_CLAIM", "FINISH"], output)
    with pytest.raises(AgentRunError) as error:
        agent.review_claim(CLAIM_ID)
    assert error.value.failure_kind == "GENERATION"
