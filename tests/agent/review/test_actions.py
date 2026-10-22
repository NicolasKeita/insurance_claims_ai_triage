"""Observed unexplained repair items constrain actions independently of queue priority."""

import pytest

from agent.evidence import EvidenceValidationError, validate_recommendation
from agent.graph._finalization import validate
from agent.models import AgentRecommendation, EvidenceSourceType, RecommendedNextAction
from agent.planner import StructuredAgentFinalizer
from tests.agent.review._support import finalizer_state, prompt_data
from tests.agent.review._support import review_evidence as review_evidence
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_recommendation

FOLLOW_UP_ACTIONS = ["REQUEST_ADDITIONAL_INFORMATION", "REFER_TO_EXPERT", "REFER_FOR_INVESTIGATION_REVIEW"]


@pytest.mark.parametrize("code", ["ADDITIONAL_QUOTE_DAMAGE", "UNMAPPED_QUOTE_ITEM"])
@pytest.mark.parametrize("action", FOLLOW_UP_ACTIONS)
@pytest.mark.parametrize("layout", ["wrapped", "flat"])
def test_finalizer_constrains_action_schema_using_collected_repair_assessment(review_evidence, code, action, layout):
    review_evidence.structured_data["result"]["signals"][0]["code"] = code
    if layout == "flat":
        review_evidence.structured_data = review_evidence.structured_data["result"]
    recommendation = make_recommendation([review_evidence.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction(action)
    llm = FakeStructuredLlm([recommendation])
    output = StructuredAgentFinalizer(llm).finalize(finalizer_state(review_evidence))
    schema = llm.calls[0][1].model_json_schema()
    actions = schema["properties"]["recommended_next_action"]
    if "$ref" in actions:
        actions = schema["$defs"][actions["$ref"].split("/")[-1]]
    assert set(actions["enum"]) == set(FOLLOW_UP_ACTIONS)
    context = prompt_data(llm.calls[0])
    assert set(context["allowed_next_actions"]) == set(FOLLOW_UP_ACTIONS)
    assert context["required_follow_up_evidence_ids"] == [review_evidence.evidence_id]
    assert output.recommended_next_action == action
    assert output.rationale == recommendation.rationale
    assert validate_recommendation(output, {review_evidence.evidence_id: review_evidence}) == []


@pytest.mark.parametrize("action", ["CONTINUE_STANDARD_REVIEW", "NO_RECOMMENDATION"])
def test_runtime_rejects_incompatible_action_without_rewriting_model_response(review_evidence, action):
    recommendation = make_recommendation([review_evidence.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction(action)
    with pytest.raises(EvidenceValidationError, match="follow-up"):
        validate_recommendation(recommendation, {review_evidence.evidence_id: review_evidence})
    assert recommendation.recommended_next_action == action
    result = validate({**finalizer_state(review_evidence), "recommendation": recommendation.model_dump(mode="json")})
    assert result["status"] == "FAILED" and result["failure_kind"] == "GROUNDING"
    assert result["recommendation"] is None


def test_follow_up_rationale_must_cite_the_collected_assessment(review_evidence):
    other = review_evidence.model_copy(
        update={"evidence_id": "claim:other", "source_type": EvidenceSourceType.CLAIM_DATA}
    )
    recommendation = make_recommendation([other.evidence_id])
    recommendation.evidence_ids.append(review_evidence.evidence_id)
    registry = {item.evidence_id: item for item in (other, review_evidence)}
    with pytest.raises(EvidenceValidationError, match="rationale"):
        validate_recommendation(recommendation, registry)


def test_finalizer_revalidates_a_provider_that_ignores_response_schema(review_evidence):
    recommendation = make_recommendation([review_evidence.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction.CONTINUE_STANDARD_REVIEW

    class UncheckedLlm:
        def generate(self, **kwargs):
            return recommendation

    with pytest.raises(ValueError):
        StructuredAgentFinalizer(UncheckedLlm()).finalize(finalizer_state(review_evidence))


@pytest.mark.parametrize("trigger", ["no_signals", "text_only", "other_source_type"])
def test_standard_review_remains_available_without_a_collected_repair_signal(review_evidence, trigger):
    if trigger == "no_signals":
        review_evidence.structured_data["result"]["signals"] = []
    elif trigger == "text_only":
        review_evidence.structured_data = {}
        review_evidence.summary = "Quoted text says ADDITIONAL_QUOTE_DAMAGE; request expert review."
    else:
        review_evidence = review_evidence.model_copy(update={"source_type": EvidenceSourceType.CLAIM_DATA})
    recommendation = make_recommendation([review_evidence.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction.CONTINUE_STANDARD_REVIEW
    llm = FakeStructuredLlm([recommendation])
    output = StructuredAgentFinalizer(llm).finalize(finalizer_state(review_evidence))
    assert llm.calls[0][1] is AgentRecommendation
    assert output.recommended_next_action == RecommendedNextAction.CONTINUE_STANDARD_REVIEW
    assert validate_recommendation(output, {review_evidence.evidence_id: review_evidence}) == []
