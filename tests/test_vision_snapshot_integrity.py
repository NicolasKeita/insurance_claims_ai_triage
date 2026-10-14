"""Audit snapshots must preserve source detections, identities and conclusions."""

import copy
from uuid import uuid4

import pytest
from pydantic import ValidationError

from persistence.vision_assessments import vision_assessment_to_row
from test_multimodal_investigation import make_visual_assessment
from vision.geometry import PolygonGeometryError, validated_polygon_geometry
from vision.models import ClaimMultimodalVisionAssessment, NormalizedPolygon


@pytest.mark.parametrize("mutation", [
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
])
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


def test_forged_containment_is_rejected_even_when_primary_reference_is_also_changed():
    payload = make_visual_assessment().model_dump(round_trip=True)
    evidence = payload["images"][0]["evidence"][0]
    evidence["associations"][0]["damage_containment_ratio"] = 0.7
    evidence["primary_association"]["damage_containment_ratio"] = 0.7
    evidence["containment_ratio"] = 0.7
    with pytest.raises(ValidationError, match="preserved areas"):
        ClaimMultimodalVisionAssessment.model_validate(payload)


def points(x_min, y_min, x_max, y_max):
    return [{"x": x_min, "y": y_min}, {"x": x_max, "y": y_min},
            {"x": x_max, "y": y_max}, {"x": x_min, "y": y_max}]


def test_polygon_representation_preserves_holes_and_disconnected_components():
    polygon = NormalizedPolygon(
        points=points(0, 0, 0.5, 0.5), holes=[points(0.1, 0.1, 0.2, 0.2)],
        components=[NormalizedPolygon(points=points(0.8, 0.8, 1, 1))],
    )
    geometry = validated_polygon_geometry(polygon)
    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == pytest.approx(0.25 - 0.01 + 0.04)
    assert len(geometry.geoms[0].interiors) == 1
    assert NormalizedPolygon.model_validate(polygon.model_dump(round_trip=True)) == polygon


def test_overlapping_components_and_hole_outside_exterior_are_rejected():
    for polygon in [
        NormalizedPolygon(points=points(0, 0, 0.5, 0.5), components=[NormalizedPolygon(points=points(0.2, 0.2, 0.8, 0.8))]),
        NormalizedPolygon(points=points(0, 0, 0.5, 0.5), holes=[points(0.6, 0.6, 0.7, 0.7)]),
    ]:
        with pytest.raises(PolygonGeometryError, match="Invalid polygon topology"):
            validated_polygon_geometry(polygon)
