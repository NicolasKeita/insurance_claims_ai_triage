"""Collected visual non-detections remain uncertainties independently of LLM wording."""

import pytest

from agent.graph._finalization import validate
from agent.models import EvidenceSourceType
from tests.agent.review._support import finalizer_state
from tests.infrastructure.support.agent_fakes import make_evidence, make_recommendation


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("model_uncertainty_count", [0, 20])
def test_visual_non_detection_survives_empty_or_full_model_uncertainties(wrapped, model_uncertainty_count):
    data = {"consistency": {"declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"]}}
    evidence = make_evidence("vision:stored").model_copy(
        update={
            "source_type": EvidenceSourceType.VISUAL_ASSESSMENT,
            "structured_data": {"result": data} if wrapped else data,
        }
    )
    recommendation = make_recommendation([evidence.evidence_id])
    recommendation.recommended_next_action = "CONTINUE_STANDARD_REVIEW"
    recommendation.uncertainties = [f"Model uncertainty {number}" for number in range(model_uncertainty_count)]

    result = validate({**finalizer_state(evidence), "recommendation": recommendation.model_dump(mode="json")})

    assert result["status"] == "COMPLETED"
    output = result["recommendation"]
    assert output["recommended_next_action"] == "CONTINUE_STANDARD_REVIEW"
    assert output["uncertainties"][0] == (
        "Declared damage was not visually confirmed (LEFT_HEADLIGHT); non-detection is not evidence of absence."
    )
    assert len(output["uncertainties"]) == min(20, model_uncertainty_count + 1)
    if model_uncertainty_count:
        assert "Model uncertainty 0" in output["uncertainties"]


@pytest.mark.parametrize(
    "source_type, consistency",
    [
        (EvidenceSourceType.CLAIM_DATA, {"declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"]}),
        (EvidenceSourceType.HISTORY, {"declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"]}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"visually_supported_declared_damage": ["LEFT_HEADLIGHT"]}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"declared_damage_not_visually_confirmed": []}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"declared_damage_not_visually_confirmed": ["invented part"]}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, {"declared_damage_not_visually_confirmed": "LEFT_HEADLIGHT"}),
        (EvidenceSourceType.VISUAL_ASSESSMENT, None),
    ],
)
def test_only_typed_visual_non_detections_add_application_uncertainty(source_type, consistency):
    evidence = make_evidence("assessment:stored").model_copy(
        update={"source_type": source_type, "structured_data": {"consistency": consistency}}
    )
    recommendation = make_recommendation([evidence.evidence_id])
    recommendation.uncertainties = []

    result = validate({**finalizer_state(evidence), "recommendation": recommendation.model_dump(mode="json")})

    assert result["status"] == "COMPLETED"
    assert result["recommendation"]["uncertainties"] == []
