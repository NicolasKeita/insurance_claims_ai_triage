"""Only collected assessment facts can require a follow-up action."""

import pytest

from agent.evidence import validate_recommendation
from agent.models import EvidenceSourceType, RecommendedNextAction
from agent.planner import StructuredAgentFinalizer
from agent.planner._responses import recommendation_constraints
from agent.review_policy import follow_up_sources
from tests.agent.review._support import finalizer_state, prompt_data
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_evidence, make_recommendation


def assessment_evidence(source_type, data):
    return make_evidence("assessment:stored").model_copy(update={"source_type": source_type, "structured_data": data})


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_positive_anomaly_supports_registered_and_flat_assessment_formats(wrapped, registered):
    data = {"is_anomalous": True, "anomaly_score": 0.7}
    if registered:
        data = {"assessment": data, "registered_model": "offline"}
    if wrapped:
        data = {"result": data}
    item = assessment_evidence(EvidenceSourceType.ANOMALY_ASSESSMENT, data)
    registry = {item.evidence_id: item}
    assert follow_up_sources(registry) == [item.evidence_id]
    assert recommendation_constraints(registry)[1]["required_follow_up_evidence_ids"] == [item.evidence_id]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "source_type, data",
    [
        (EvidenceSourceType.TRIAGE_ASSESSMENT, {"recommended_workflow": "STANDARD"}),
        (EvidenceSourceType.TRIAGE_ASSESSMENT, {"recommended_workflow": "FAST_TRACK"}),
        (EvidenceSourceType.TRIAGE_ASSESSMENT, {"signals": [{"code": "ANOMALOUS_PATTERN"}]}),
        (EvidenceSourceType.TRIAGE_ASSESSMENT, {"is_anomalous": True}),
        (EvidenceSourceType.ANOMALY_ASSESSMENT, {"is_anomalous": False}),
        (EvidenceSourceType.ANOMALY_ASSESSMENT, {"assessment": {"is_anomalous": False}}),
        (EvidenceSourceType.ANOMALY_ASSESSMENT, {"is_anomalous": "true"}),
        (EvidenceSourceType.ANOMALY_ASSESSMENT, {"is_anomalous": 1}),
        (EvidenceSourceType.ANOMALY_ASSESSMENT, {"recommended_workflow": "EXPERT_REVIEW"}),
        (EvidenceSourceType.INVESTIGATION_ASSESSMENT, {"recommended_workflow": "EXPERT_REVIEW"}),
        (EvidenceSourceType.INVESTIGATION_ASSESSMENT, {"is_anomalous": True}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"visually_supported_declared_damage": ["HOOD"]}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"visually_supported_quote_damage": ["HOOD"]}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"declared_damage_not_visually_confirmed": ["HOOD"]}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"visual_damage_unmapped": [{"type": "SCRATCH"}]}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"additional_visual_damage": []}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"additional_visual_damage": ["invented part"]}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": {"additional_visual_damage": "HOOD"}}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"consistency": None}),
    ],
)
def test_nontriggering_assessments_keep_routine_review_available(source_type, data, wrapped):
    item = assessment_evidence(source_type, {"result": data} if wrapped else data)
    registry = {item.evidence_id: item}
    assert follow_up_sources(registry) == []
    recommendation = make_recommendation([item.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction.CONTINUE_STANDARD_REVIEW
    llm = FakeStructuredLlm([recommendation])
    output = StructuredAgentFinalizer(llm).finalize(finalizer_state(item))
    assert output == recommendation
    assert "CONTINUE_STANDARD_REVIEW" in prompt_data(llm.calls[0])["allowed_next_actions"]
    assert validate_recommendation(output, registry) == []


@pytest.mark.parametrize(
    "source_type",
    [
        EvidenceSourceType.CLAIM_DATA,
        EvidenceSourceType.HISTORY,
        EvidenceSourceType.VISUAL_ASSESSMENT,
        EvidenceSourceType.SIMILAR_CLAIM,
        EvidenceSourceType.POLICY_CHUNK,
        EvidenceSourceType.PROCEDURE_CHUNK,
    ],
)
def test_follow_up_looking_data_in_other_source_types_cannot_trigger(source_type):
    item = assessment_evidence(
        source_type,
        {
            "recommended_workflow": "EXPERT_REVIEW",
            "is_anomalous": True,
            "signals": [{"code": "ANOMALOUS_PATTERN"}],
        },
    )
    assert follow_up_sources({item.evidence_id: item}) == []


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("source_type", [EvidenceSourceType.CLAIM_DATA, EvidenceSourceType.HISTORY])
def test_visual_looking_fields_in_other_sources_cannot_trigger_follow_up(source_type, wrapped):
    data = {"consistency": {"additional_visual_damage": ["HOOD"]}}
    item = assessment_evidence(source_type, {"result": data} if wrapped else data)
    assert follow_up_sources({item.evidence_id: item}) == []


def test_unavailable_assessments_do_not_create_follow_up_requirements():
    claim = make_evidence()
    state = finalizer_state(claim)
    state["observations"] = [
        {"tool_name": tool, "status": "NOT_AVAILABLE", "message": "No stored assessment is available"}
        for tool in ("GET_TRIAGE", "GET_ANOMALY", "GET_INVESTIGATION")
    ]
    recommendation = make_recommendation([claim.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction.CONTINUE_STANDARD_REVIEW
    llm = FakeStructuredLlm([recommendation])
    assert StructuredAgentFinalizer(llm).finalize(state) == recommendation
    assert prompt_data(llm.calls[0])["required_follow_up_evidence_ids"] == []


def test_every_triggering_assessment_is_listed_once_in_registry_order():
    triage = assessment_evidence(EvidenceSourceType.TRIAGE_ASSESSMENT, {"recommended_workflow": "EXPERT_REVIEW"})
    anomaly = assessment_evidence(EvidenceSourceType.ANOMALY_ASSESSMENT, {"is_anomalous": True}).model_copy(
        update={"evidence_id": "anomaly:stored"}
    )
    investigation = assessment_evidence(
        EvidenceSourceType.INVESTIGATION_ASSESSMENT,
        {"signals": [{"code": "ANOMALOUS_PATTERN"}, {"code": "EXTREME_ANOMALY_PERCENTILE"}]},
    ).model_copy(update={"evidence_id": "investigation:stored"})
    registry = {item.evidence_id: item for item in (triage, anomaly, investigation)}
    assert follow_up_sources(registry) == list(registry)
