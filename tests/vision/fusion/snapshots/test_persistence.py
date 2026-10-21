"""Test persistence."""

import copy
from uuid import uuid4

import pytest
from pydantic import ValidationError

from persistence.vision_assessments import vision_assessment_to_row
from tests.investigation.multimodal._assessment import make_visual_assessment
from vision.models import (
    ClaimMultimodalVisionAssessment,
)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["damage_model_metadata"].update(version="different-damage-version"),
        lambda payload: payload["part_model_metadata"].update(version="different-part-version"),
        lambda payload: payload["fusion_policy"].update(ambiguity_margin=0.02),
        lambda payload: payload["images"][0]["part_assessment"].update(filename="different-image.png"),
        lambda payload: payload["images"][0]["part_assessment"].update(width=50),
        lambda payload: payload["images"][0]["damage_assessment"].update(detections=[]),
        lambda payload: payload["images"][0].update(evidence=[]),
        lambda payload: payload["images"][0]["evidence"].append(copy.deepcopy(payload["images"][0]["evidence"][0])),
        lambda payload: payload["images"][0]["part_assessment"]["detections"][0].update(confidence=0.95),
        lambda payload: payload["consistency"].update(additional_visual_damage=[]),
        lambda payload: payload["consistency"].update(visually_supported_declared_damage=["FRONT_BUMPER"]),
        lambda payload: payload["consistency"].update(visually_supported_quote_damage=[]),
        lambda payload: payload["consistency"].update(declared_damage_not_visually_confirmed=[]),
    ],
)
def test_tampered_domain_graph_is_rejected_before_persistence(mutation):
    payload = make_visual_assessment().model_dump(round_trip=True)
    mutation(payload)
    with pytest.raises(ValidationError):
        ClaimMultimodalVisionAssessment.model_validate(payload)


def test_mutating_nested_lists_cannot_persist_phantom_evidence():
    assessment = make_visual_assessment()
    assessment.images[0].damage_assessment.detections.clear()
    with pytest.raises(ValidationError, match="exactly one"):
        vision_assessment_to_row(uuid4(), assessment)


def test_known_image_reference_mismatch_is_rejected():
    payload = make_visual_assessment().model_dump(round_trip=True)
    payload["images"][0]["damage_assessment"]["asset_reference"] = "actual/images/another.jpg"
    with pytest.raises(ValidationError, match="asset reference"):
        ClaimMultimodalVisionAssessment.model_validate(payload)
