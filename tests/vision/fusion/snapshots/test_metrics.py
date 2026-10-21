"""Test metrics."""

import pytest
from pydantic import ValidationError

from tests.investigation.multimodal._assessment import make_visual_assessment
from vision.models import ClaimMultimodalVisionAssessment, ImageVisualAssessment


def test_forged_containment_is_rejected_even_when_primary_reference_is_also_changed():
    payload = make_visual_assessment().model_dump(round_trip=True)
    evidence = payload["images"][0]["evidence"][0]
    evidence["associations"][0]["damage_containment_ratio"] = 0.7
    evidence["primary_association"]["damage_containment_ratio"] = 0.7
    evidence["containment_ratio"] = 0.7
    with pytest.raises(ValidationError, match="preserved areas"):
        ClaimMultimodalVisionAssessment.model_validate(payload)


def test_internally_consistent_forged_areas_cannot_replace_observed_mask_overlap():
    """
    These scalars agree with one another, but not with the preserved masks.
    """
    payload = make_visual_assessment().model_dump(round_trip=True)
    evidence = payload["images"][0]["evidence"][0]
    for association in (evidence["associations"][0], evidence["primary_association"]):
        association.update(intersection_area=0.007, damage_containment_ratio=0.7, iou=0.007 / 0.253)
    evidence["containment_ratio"] = 0.7
    with pytest.raises(ValidationError, match="source masks"):
        ClaimMultimodalVisionAssessment.model_validate(payload)


@pytest.mark.parametrize("field,value", [("association_method", "BBOX_FALLBACK"), ("eligible", False)])
def test_association_method_and_eligibility_cannot_be_forged(field, value):
    """
    Two complete overlapping candidates are deliberately ambiguous.
    """
    from tests.vision.fusion.association._support import damage, fused, part

    payload = fused([damage()], [part(), part()]).model_dump(round_trip=True)
    payload["evidence"][0]["associations"][0][field] = value
    with pytest.raises(ValidationError, match="method or eligibility"):
        ImageVisualAssessment.model_validate(payload)
