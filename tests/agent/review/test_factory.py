"""Production wiring collects stored investigation and triage before the model can finish."""

from dataclasses import replace

import pytest

from agent.factory import create_claims_agent_service
from claims.enums import TriageWorkflow
from tests.agent.review._support import prompt_data
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import FakeRunStore, FakeStructuredLlm, make_recommendation, make_step

CORE_AND_VISUAL_READS = ["GET_CLAIM", "GET_INVESTIGATION", "GET_TRIAGE", "GET_VISUAL_ASSESSMENT"]


@pytest.mark.parametrize("missing", [None, "investigation", "triage"])
def test_production_agent_collects_core_assessments_even_when_model_immediately_finishes(
    monkeypatch, stored_review, missing
):
    claim, factory, calls, records = stored_review
    investigation_id = f"investigation:{records['investigation'].id}"
    if missing is not None:
        records[missing] = None
    recommendation = make_recommendation(
        [f"claim:{claim.claim_id}"] + ([] if missing == "investigation" else [investigation_id])
    )
    llm = FakeStructuredLlm([make_step("FINISH"), recommendation])
    store = FakeRunStore()
    monkeypatch.setenv("CLAIMS_LLM_MODEL", "fake-local")
    monkeypatch.setattr("agent.factory._llm", lambda _: llm)
    monkeypatch.setattr("agent.factory.AgentRunStore", lambda _: store)
    result = create_claims_agent_service(factory).review_claim(claim.claim_id)
    assert result.status == "COMPLETED"
    assert [tool.value for tool in result.tools_used] == CORE_AND_VISUAL_READS
    assert calls.count(("investigation", claim.claim_id)) == 1
    assert calls.count(("triage", claim.claim_id)) == 1
    assert result.tool_call_count == result.iteration_count == 4
    assert store.results == [result] and factory.open_count == 0
    assert len(llm.calls) == 2
    final_context = prompt_data(llm.calls[-1])
    if missing == "investigation":
        assert all(item["evidence_id"] != investigation_id for item in final_context["evidence"])
        assert (
            "GET_INVESTIGATION: No stored investigation assessment is available" in result.recommendation.uncertainties
        )
    else:
        evidence = next(item for item in final_context["evidence"] if item["evidence_id"] == investigation_id)
        assert "ADDITIONAL_QUOTE_DAMAGE" in evidence["summary"]
        assert "HOOD" in evidence["structured_data"]["signal_details"][0]["facts"]
    if missing == "triage":
        assert "GET_TRIAGE: No stored triage assessment is available" in result.recommendation.uncertainties
        assert all(item["source_type"] != "TRIAGE_ASSESSMENT" for item in final_context["evidence"])


@pytest.mark.parametrize("workflow", ["EXPERT_REVIEW", "INVESTIGATION"])
def test_triage_requires_follow_up_even_when_investigation_priority_is_normal(monkeypatch, stored_review, workflow):
    claim, factory, _, records = stored_review
    records["investigation"] = replace(
        records["investigation"], result=records["investigation"].result.model_copy(update={"signals": ()})
    )
    records["triage"] = replace(
        records["triage"],
        result=records["triage"].result.model_copy(update={"recommended_workflow": TriageWorkflow(workflow)}),
    )
    triage_id = f"triage:{records['triage'].id}"
    recommendation = make_recommendation([triage_id])
    llm = FakeStructuredLlm([make_step("FINISH"), recommendation])
    monkeypatch.setenv("CLAIMS_LLM_MODEL", "fake-local")
    monkeypatch.setattr("agent.factory._llm", lambda _: llm)
    monkeypatch.setattr("agent.factory.AgentRunStore", lambda _: FakeRunStore())
    result = create_claims_agent_service(factory).review_claim(claim.claim_id)
    assert result.recommendation.recommended_next_action == recommendation.recommended_next_action
    assert result.recommendation.rationale == recommendation.rationale
    assert "GET_VISUAL_ASSESSMENT: No stored visual assessment is available" in result.recommendation.uncertainties
    assert prompt_data(llm.calls[-1])["required_follow_up_evidence_ids"] == [triage_id]
    assert "CONTINUE_STANDARD_REVIEW" not in prompt_data(llm.calls[-1])["allowed_next_actions"]
