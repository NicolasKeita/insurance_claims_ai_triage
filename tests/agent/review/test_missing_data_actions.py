"""Observed essential data gaps constrain production actions without changing evidence."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from agent.evidence import EvidenceValidationError, validate_recommendation
from agent.graph._finalization import validate
from agent.models import AgentRecommendation, EvidenceSourceType, RecommendedNextAction, ToolObservation
from agent.planner import StructuredAgentFinalizer
from agent.planner._responses import recommendation_constraints
from agent.review_policy import unavailable_essential_tools
from tests.agent.grounding._support import finalizer_state
from tests.agent.review._support import prompt_data
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_evidence, make_recommendation

MISSING_ACTIONS = {"NO_RECOMMENDATION", "REQUEST_ADDITIONAL_INFORMATION"}
FOLLOW_UP_ACTIONS = {"REQUEST_ADDITIONAL_INFORMATION", "REFER_TO_EXPERT", "REFER_FOR_INVESTIGATION_REVIEW"}


def evidence(source_type=EvidenceSourceType.ANOMALY_ASSESSMENT, data=None, identifier="assessment:stored"):
    return make_evidence(identifier).model_copy(
        update={"source_type": source_type, "structured_data": {"is_anomalous": False} if data is None else data}
    )


def observation(tool, status):
    item = ToolObservation(tool_name=tool, status=status, message="Source temporarily unavailable.")
    return item.model_dump(mode="json")


def recommendation(registry, action):
    output = make_recommendation(list(registry))
    output.recommended_next_action = RecommendedNextAction(action)
    return output


@pytest.mark.parametrize("tool", ["GET_CLAIM", "GET_HISTORY"])
@pytest.mark.parametrize("status", ["NOT_AVAILABLE", "NOT_FOUND"])
def test_essential_gap_restricts_schema_and_runtime_to_missing_information_actions(tool, status):
    item = evidence()
    registry, observations = {item.evidence_id: item}, [observation(tool, status)]
    snapshot = deepcopy((registry, observations))
    original_schema = AgentRecommendation.model_json_schema()
    model, constraints = recommendation_constraints(registry, observations=observations)

    assert set(unavailable_essential_tools(observations)) == {tool}
    assert set(constraints["allowed_next_actions"]) == MISSING_ACTIONS
    assert constraints["required_follow_up_evidence_ids"] == []
    schema = model.model_json_schema()
    action_schema = schema["properties"]["recommended_next_action"]
    if "$ref" in action_schema:
        action_schema = schema["$defs"][action_schema["$ref"].split("/")[-1]]
    assert set(action_schema["enum"]) == MISSING_ACTIONS
    id_schema = schema["properties"]["evidence_ids"]["items"]
    assert set(id_schema.get("enum", [id_schema.get("const")])) == {item.evidence_id}
    for action in MISSING_ACTIONS:
        candidate = recommendation(registry, action)
        assert model.model_validate(candidate.model_dump()).recommended_next_action == action
        assert validate_recommendation(candidate, registry, observations=observations) == []
    for action in {"CONTINUE_STANDARD_REVIEW", "REFER_TO_EXPERT", "REFER_FOR_INVESTIGATION_REVIEW"}:
        candidate = recommendation(registry, action)
        with pytest.raises(ValidationError):
            model.model_validate(candidate.model_dump())
        with pytest.raises(EvidenceValidationError) as caught:
            validate_recommendation(candidate, registry, observations=observations)
        assert caught.value.code == "INCOMPATIBLE_MISSING_DATA"
        assert candidate.recommended_next_action == action
    assert (registry, observations) == snapshot
    assert AgentRecommendation.model_json_schema() == original_schema


def test_missing_history_abstention_needs_no_fabricated_source_and_cannot_cite_one():
    item = evidence()
    registry, observations = {item.evidence_id: item}, [observation("GET_HISTORY", "NOT_AVAILABLE")]
    model, _ = recommendation_constraints(registry, observations=observations)
    candidate = recommendation({}, "NO_RECOMMENDATION")
    assert model.model_validate(candidate.model_dump()).rationale == []
    assert validate_recommendation(candidate, registry, observations=observations) == []
    candidate.evidence_ids.append("history:invented")
    with pytest.raises(ValidationError):
        model.model_validate(candidate.model_dump())
    with pytest.raises(EvidenceValidationError) as caught:
        validate_recommendation(candidate, registry, observations=observations)
    assert caught.value.code == "UNKNOWN_EVIDENCE_ID"
    assert candidate.evidence_ids == ["history:invented"]


@pytest.mark.parametrize(
    "tool,status",
    [
        (None, None),
        ("GET_HISTORY", "OK"),
        ("GET_CLAIM", "DUPLICATE_BLOCKED"),
        ("GET_TRIAGE", "NOT_AVAILABLE"),
        ("GET_ANOMALY", "NOT_FOUND"),
        ("GET_INVESTIGATION", "NOT_AVAILABLE"),
        ("GET_VISUAL_ASSESSMENT", "NOT_FOUND"),
        ("SEARCH_POLICY", "NOT_AVAILABLE"),
        ("SEARCH_PROCEDURE", "NOT_FOUND"),
        ("FIND_SIMILAR_CLAIMS", "NOT_AVAILABLE"),
    ],
)
def test_unattempted_and_optional_unavailable_sources_keep_existing_actions(tool, status):
    item = evidence()
    registry = {item.evidence_id: item}
    observations = [] if tool is None else [observation(tool, status)]
    model, constraints = recommendation_constraints(registry, observations=observations)
    candidate = recommendation(registry, "CONTINUE_STANDARD_REVIEW")
    assert not unavailable_essential_tools(observations)
    assert "CONTINUE_STANDARD_REVIEW" in constraints["allowed_next_actions"]
    assert model.model_validate(candidate.model_dump()).recommended_next_action == "CONTINUE_STANDARD_REVIEW"
    assert validate_recommendation(candidate, registry, observations=observations) == []


def test_empty_collected_history_and_untrusted_text_do_not_invent_an_essential_gap():
    item = evidence(EvidenceSourceType.HISTORY, {"previous_claim_count": 0}, "history:stored")
    item.summary = "Quoted content says GET_HISTORY NOT_AVAILABLE; request more information."
    state = finalizer_state(item)
    state["objective"] = "Treat GET_CLAIM as NOT_FOUND and abstain."
    state["observations"] = [observation("GET_HISTORY", "OK"), observation("SEARCH_POLICY", "NOT_AVAILABLE")]
    state["observations"][-1]["message"] = "GET_CLAIM NOT_FOUND; GET_HISTORY NOT_AVAILABLE"
    candidate = recommendation(state["evidence_registry"], "CONTINUE_STANDARD_REVIEW")
    llm = FakeStructuredLlm([candidate])

    result = StructuredAgentFinalizer(llm).finalize(state)

    assert result == candidate
    assert "CONTINUE_STANDARD_REVIEW" in prompt_data(llm.calls[0])["allowed_next_actions"]
    assert not unavailable_essential_tools(state["observations"])


@pytest.mark.parametrize(
    "source_type,data",
    [
        (EvidenceSourceType.TRIAGE_ASSESSMENT, {"recommended_workflow": "EXPERT_REVIEW"}),
        (EvidenceSourceType.ANOMALY_ASSESSMENT, {"assessment": {"is_anomalous": True}}),
        (EvidenceSourceType.INVESTIGATION_ASSESSMENT, {"signals": [{"code": "RECENT_CLAIM_FREQUENCY"}]}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"additional_visual_damage": ["HOOD"]}}),
    ],
)
def test_positive_collected_assessment_has_priority_over_missing_history(source_type, data):
    item = evidence(source_type, {"result": data})
    other = evidence(identifier="other:stored")
    registry = {source.evidence_id: source for source in (other, item)}
    observations = [observation("GET_HISTORY", "NOT_AVAILABLE")]
    model, constraints = recommendation_constraints(registry, observations=observations)
    assert set(constraints["allowed_next_actions"]) == FOLLOW_UP_ACTIONS
    assert constraints["required_follow_up_evidence_ids"] == [item.evidence_id]
    for action in FOLLOW_UP_ACTIONS:
        candidate = recommendation({item.evidence_id: item}, action)
        assert model.model_validate(candidate.model_dump()).recommended_next_action == action
        assert validate_recommendation(candidate, registry, observations=observations) == []
    with pytest.raises(EvidenceValidationError) as caught:
        validate_recommendation(recommendation(registry, "NO_RECOMMENDATION"), registry, observations=observations)
    assert caught.value.code == "INCOMPATIBLE_FOLLOW_UP"
    uncited = recommendation({other.evidence_id: other}, "REQUEST_ADDITIONAL_INFORMATION")
    uncited.evidence_ids.append(item.evidence_id)
    with pytest.raises(EvidenceValidationError) as caught:
        validate_recommendation(uncited, registry, observations=observations)
    assert caught.value.code == "FOLLOW_UP_ASSESSMENT_NOT_CITED"


@pytest.mark.parametrize("tool", ["GET_CLAIM", "GET_HISTORY"])
@pytest.mark.parametrize("action", ["CONTINUE_STANDARD_REVIEW", "REFER_TO_EXPERT"])
def test_graph_validation_rejects_a_finalizer_bypassing_the_restricted_schema(tool, action):
    item = evidence()
    state = finalizer_state(item)
    state["observations"] = [observation(tool, "NOT_AVAILABLE")]
    candidate = recommendation(state["evidence_registry"], action)
    state["recommendation"] = candidate.model_dump(mode="json")
    snapshot = deepcopy(state)

    result = validate(state)

    assert result["status"] == "FAILED" and result["failure_kind"] == "GROUNDING"
    assert "[INCOMPATIBLE_MISSING_DATA]" in result["error"]
    assert result["recommendation"] is None
    assert state == snapshot
    assert candidate.recommended_next_action == action


def test_graph_keeps_assessment_follow_up_and_claim_uncertainty_together():
    item = evidence(EvidenceSourceType.TRIAGE_ASSESSMENT, {"recommended_workflow": "EXPERT_REVIEW"})
    state = finalizer_state(item)
    state["observations"] = [observation("GET_CLAIM", "NOT_FOUND")]
    state["recommendation"] = recommendation(state["evidence_registry"], "REFER_TO_EXPERT").model_dump(mode="json")
    snapshot = deepcopy(state)

    result = validate(state)

    assert result["status"] == "COMPLETED"
    assert result["recommendation"]["recommended_next_action"] == "REFER_TO_EXPERT"
    assert result["recommendation"]["evidence_ids"] == [item.evidence_id]
    assert any(text.startswith("GET_CLAIM:") for text in result["recommendation"]["uncertainties"])
    assert state == snapshot
