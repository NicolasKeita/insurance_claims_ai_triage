"""Production wiring supports a service-scoped historical read outage."""

import pytest
from sqlalchemy.exc import OperationalError

from agent.factory import create_claims_agent_service
from agent.models import ObservationStatus
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import FakeRunStore, FakeStructuredLlm, make_recommendation, make_step


@pytest.mark.parametrize("require_history", [False, True])
def test_factory_scopes_history_io_and_keeps_the_required_read_optional(monkeypatch, stored_review, require_history):
    claim, factory, calls, records = stored_review
    investigation_id = f"investigation:{records['investigation'].id}"
    llm = FakeStructuredLlm([make_step("FINISH"), make_recommendation([f"claim:{claim.claim_id}", investigation_id])])
    store = FakeRunStore()
    history_attempts = []

    def unavailable_history_sessions():
        history_attempts.append("connection attempted")
        raise OperationalError(None, None, RuntimeError("controlled isolated historical outage"))

    monkeypatch.setenv("CLAIMS_LLM_MODEL", "fake-local")
    monkeypatch.setattr("agent.factory._llm", lambda _: llm)
    monkeypatch.setattr("agent.factory.AgentRunStore", lambda _: store)
    options = {"history_session_factory": unavailable_history_sessions}
    if require_history:
        options["require_history"] = True
    service = create_claims_agent_service(factory, **options)

    result = service.review_claim(claim.claim_id)

    expected_tools = ["GET_CLAIM", "GET_INVESTIGATION", "GET_TRIAGE"]
    if require_history:
        expected_tools.append("GET_HISTORY")
    expected_tools.append("GET_VISUAL_ASSESSMENT")
    assert service.tools.history_session_factory is unavailable_history_sessions
    assert [tool.value for tool in result.tools_used] == expected_tools
    assert result.status == "COMPLETED"
    assert result.tool_call_count == result.iteration_count == len(expected_tools)
    assert len(history_attempts) == int(require_history)
    assert ("history", claim.claim_id) not in calls
    assert calls.count(("investigation", claim.claim_id)) == 1
    assert calls.count(("triage", claim.claim_id)) == 1
    assert len(llm.calls) == 2
    assert store.results == [result] and factory.open_count == 0
    if require_history:
        history_step = next(step for step in result.trace if step.tool_invoked == "GET_HISTORY")
        assert history_step.observation_status == ObservationStatus.NOT_AVAILABLE
        assert "GET_HISTORY: Historical claim data is temporarily unavailable" in result.recommendation.uncertainties
        assert not any(item.source_type == "HISTORY" for item in result.evidence)
