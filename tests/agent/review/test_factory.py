"""Production wiring must collect stored investigation before a model can finish."""

import pytest

from agent.factory import create_claims_agent_service
from tests.agent.review._support import prompt_data
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import FakeRunStore, FakeStructuredLlm, make_recommendation, make_step


@pytest.mark.parametrize("missing", [False, True])
def test_production_agent_collects_investigation_even_when_model_immediately_finishes(
    monkeypatch, stored_review, missing
):
    claim, factory, calls, records = stored_review
    investigation_id = f"investigation:{records['investigation'].id}"
    if missing:
        records["investigation"] = None
    recommendation = make_recommendation([f"claim:{claim.claim_id}"] + ([] if missing else [investigation_id]))
    llm = FakeStructuredLlm([make_step("FINISH"), recommendation])
    store = FakeRunStore()
    monkeypatch.setenv("CLAIMS_LLM_MODEL", "fake-local")
    monkeypatch.setattr("agent.factory._llm", lambda _: llm)
    monkeypatch.setattr("agent.factory.AgentRunStore", lambda _: store)
    result = create_claims_agent_service(factory).review_claim(claim.claim_id)
    assert result.status == "COMPLETED"
    assert [tool.value for tool in result.tools_used] == ["GET_CLAIM", "GET_INVESTIGATION"]
    assert calls.count(("investigation", claim.claim_id)) == 1
    assert result.tool_call_count == result.iteration_count == 2
    assert store.results == [result] and factory.open_count == 0
    assert len(llm.calls) == 2
    final_context = prompt_data(llm.calls[-1])
    if missing:
        assert all(item["evidence_id"] != investigation_id for item in final_context["evidence"])
        assert (
            "GET_INVESTIGATION: No stored investigation assessment is available" in result.recommendation.uncertainties
        )
    else:
        evidence = next(item for item in final_context["evidence"] if item["evidence_id"] == investigation_id)
        assert "ADDITIONAL_QUOTE_DAMAGE" in evidence["summary"]
        assert "HOOD" in evidence["structured_data"]["signal_details"][0]["facts"]
