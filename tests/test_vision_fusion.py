"""Dataset/GPU-free geometry and domain tests, including the explicit IoU trap."""

import pytest
from pydantic import ValidationError

from claims.enums import DamageType
from vision.fusion import FusionGeometryError, calculate_overlap, fuse_image_assessments
from vision.fusion_config import VisionFusionPolicy
from vision.models import (
    AssociationMethod, AssociationStatus, ImageDamageAssessment, ImagePartAssessment,
    NormalizedBoundingBox, NormalizedPoint, NormalizedPolygon, VehiclePartDetection,
    VisualDamageDetection, VisionModelMetadata,
)
from vision.taxonomy import (
    VehiclePartType, VisualDamageType, map_vehicle_part_category_name, map_vehicle_part_to_damage,
)

METADATA = VisionModelMetadata(identifier="synthetic-model", framework="fake", version="test-only")


def rectangle(x_min, y_min, x_max, y_max):
    return NormalizedBoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def polygon_for(bounds):
    return NormalizedPolygon(points=[
        NormalizedPoint(x=bounds.x_min, y=bounds.y_min),
        NormalizedPoint(x=bounds.x_max, y=bounds.y_min),
        NormalizedPoint(x=bounds.x_max, y=bounds.y_max),
        NormalizedPoint(x=bounds.x_min, y=bounds.y_max),
    ])


def damage(bounds=None, *, confidence=0.9, mask=True):
    bounds = bounds or rectangle(0.1, 0.1, 0.2, 0.2)
    return VisualDamageDetection(
        damage_type=VisualDamageType.SCRATCH, confidence=confidence,
        bounding_box=bounds, polygon=polygon_for(bounds) if mask else None,
    )


def part(bounds=None, *, kind=VehiclePartType.HOOD, confidence=0.9, mask=True):
    bounds = bounds or rectangle(0, 0, 1, 1)
    return VehiclePartDetection(
        part_type=kind, confidence=confidence,
        bounding_box=bounds, polygon=polygon_for(bounds) if mask else None,
    )


def fused(detections, parts, *, policy=None, **part_image_changes):
    return fuse_image_assessments(
        ImageDamageAssessment(filename="synthetic.png", width=100, height=100, detections=detections),
        ImagePartAssessment(**{
            "filename": "synthetic.png", "width": 100, "height": 100,
            "detections": parts, "model_metadata": METADATA, "confidence_threshold": 0.25,
        } | part_image_changes),
        damage_model_metadata=METADATA, policy=policy,
    )


def test_iou_trap_small_damage_inside_large_part_is_assigned_by_containment():
    result = fused([damage()], [part()])
    association = result.evidence[0].primary_association
    assert association.damage_area == pytest.approx(0.01)
    assert association.part_area == pytest.approx(1.0)
    assert association.intersection_area == pytest.approx(0.01)
    assert association.damage_containment_ratio == pytest.approx(1.0)
    assert association.iou == pytest.approx(0.01)
    assert association.association_method is AssociationMethod.MASK
    assert result.evidence[0].association_status is AssociationStatus.ASSIGNED
    assert result.evidence[0].mapped_damage_type is DamageType.HOOD


@pytest.mark.parametrize("part_bounds,containment", [
    (rectangle(0.15, 0.1, 0.25, 0.2), 0.5),
    (rectangle(0.3, 0.3, 0.5, 0.5), 0.0),
])
def test_half_and_zero_overlap(part_bounds, containment):
    result = calculate_overlap(damage(), part(part_bounds))
    assert result.damage_containment_ratio == pytest.approx(containment)
    assert result.intersection_area == pytest.approx(result.damage_area * containment)


def test_masks_take_precedence_over_overlapping_boxes():
    # Both boxes cover the image, but the observed mask regions are disjoint.
    full_box = rectangle(0, 0, 1, 1)
    observation = damage(full_box).model_copy(update={"polygon": polygon_for(rectangle(0, 0, 0.2, 0.2))})
    vehicle_part = part(full_box).model_copy(update={"polygon": polygon_for(rectangle(0.8, 0.8, 1, 1))})
    overlap = calculate_overlap(observation, vehicle_part)
    assert overlap.association_method is AssociationMethod.MASK
    assert overlap.damage_containment_ratio == 0
    assert fused([observation], [vehicle_part]).evidence[0].association_status is AssociationStatus.UNASSIGNED


def test_damage_in_a_part_mask_hole_has_zero_containment_despite_the_outer_box():
    vehicle_part = part().model_copy(update={"polygon": NormalizedPolygon(
        points=polygon_for(rectangle(0, 0, 1, 1)).points,
        holes=[polygon_for(rectangle(0.05, 0.05, 0.25, 0.25)).points],
    )})
    overlap = calculate_overlap(damage(), vehicle_part)
    assert overlap.association_method is AssociationMethod.MASK
    assert overlap.damage_containment_ratio == 0
    assert overlap.part_area == pytest.approx(0.96)
    assert fused([damage()], [vehicle_part]).evidence[0].association_status is AssociationStatus.UNASSIGNED


def test_disconnected_part_component_can_support_a_contained_damage():
    vehicle_part = part().model_copy(update={"polygon": NormalizedPolygon(
        points=polygon_for(rectangle(0.7, 0.7, 1, 1)).points,
        components=[polygon_for(rectangle(0, 0, 0.3, 0.3))],
    )})
    evidence = fused([damage()], [vehicle_part]).evidence[0]
    assert evidence.association_status is AssociationStatus.ASSIGNED
    assert evidence.containment_ratio == pytest.approx(1)
    assert evidence.primary_association.part_area == pytest.approx(0.18)


@pytest.mark.parametrize("damage_mask,part_mask", [(False, True), (True, False), (False, False)])
def test_only_missing_masks_use_explicit_bbox_fallback(damage_mask, part_mask):
    overlap = calculate_overlap(damage(mask=damage_mask), part(mask=part_mask))
    assert overlap.damage_containment_ratio == pytest.approx(1)
    assert overlap.association_method is AssociationMethod.BBOX_FALLBACK


@pytest.mark.parametrize("coordinates", [
    [(0, 0), (1, 1), (0, 1), (1, 0)],
    [(0.1, 0.1), (0.9, 0.8), (0.1, 0.9), (0.8, 0.1)],
])
def test_invalid_self_crossing_polygon_rejected(coordinates):
    points = [NormalizedPoint(x=x, y=y) for x, y in coordinates]
    try:
        polygon = NormalizedPolygon(points=points)
    except ValidationError:
        return  # Step29 already rejects signed-zero area geometry.
    with pytest.raises(FusionGeometryError, match="fallback is forbidden"):
        calculate_overlap(damage().model_copy(update={"polygon": polygon}), part())


def test_step29_retraced_nonzero_ring_can_load_but_is_rejected_for_fusion():
    # Ultralytics xyn can bridge contours using a retraced edge. Preserve the
    # step29 artifact loading contract; fusion must reject its invalid topology.
    polygon = NormalizedPolygon(points=[NormalizedPoint(x=x, y=y) for x, y in [
        (0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3), (0.1, 0.1),
        (0.5, 0.1), (0.7, 0.1), (0.7, 0.3), (0.5, 0.3), (0.5, 0.1), (0.1, 0.1),
    ]])
    with pytest.raises(FusionGeometryError, match="fallback is forbidden"):
        calculate_overlap(damage().model_copy(update={"polygon": polygon}), part())


@pytest.mark.parametrize("part_mask", [True, False])
def test_invalid_geometry_never_silently_becomes_bbox_even_if_other_mask_missing(part_mask):
    invalid = NormalizedPolygon.model_construct(points=[
        NormalizedPoint(x=x, y=y) for x, y in [(0, 0), (1, 1), (0, 1), (1, 0)]
    ])
    observation = damage().model_copy(update={"polygon": invalid})
    with pytest.raises(FusionGeometryError, match="fallback is forbidden"):
        calculate_overlap(observation, part(mask=part_mask))
    with pytest.raises(FusionGeometryError):
        fused([observation], [])


def test_highest_containment_selects_primary_preserving_other_candidates():
    result = fused([damage(rectangle(0.1, 0.1, 0.3, 0.3))], [
        part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
        part(kind=VehiclePartType.HOOD),
    ])
    evidence = result.evidence[0]
    assert evidence.association_status is AssociationStatus.ASSIGNED
    assert evidence.vehicle_part is VehiclePartType.HOOD
    assert [candidate.damage_containment_ratio for candidate in evidence.associations] == pytest.approx([1, 0.8])


def test_close_eligible_candidates_remain_ambiguous_without_forced_business_mapping():
    result = fused([damage(rectangle(0.1, 0.1, 0.3, 0.3))], [
        part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
        part(rectangle(0.1, 0.1, 0.265, 0.3), kind=VehiclePartType.HOOD),
    ])
    evidence = result.evidence[0]
    assert evidence.association_status is AssociationStatus.AMBIGUOUS
    assert len(evidence.associations) == 2
    assert evidence.primary_association is None
    assert evidence.vehicle_part is None and evidence.mapped_damage_type is None


@pytest.mark.parametrize("observation,vehicle_part", [
    (damage(confidence=0.49), part()),
    (damage(), part(confidence=0.49)),
    (damage(), part(rectangle(0.15, 0.1, 0.3, 0.3))),
    (damage(), part(rectangle(0.8, 0.8, 1, 1))),
])
def test_unreliable_or_uncovered_damage_is_unassigned(observation, vehicle_part):
    evidence = fused([observation], [vehicle_part]).evidence[0]
    assert evidence.association_status is AssociationStatus.UNASSIGNED
    assert evidence.mapped_damage_type is None


def test_low_confidence_best_overlap_does_not_hide_eligible_part():
    result = fused([damage(rectangle(0.1, 0.1, 0.3, 0.3))], [
        part(kind=VehiclePartType.HOOD, confidence=0.4),
        part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
    ])
    assert result.evidence[0].vehicle_part is VehiclePartType.FRONT_BUMPER
    assert len(result.evidence[0].associations) == 2
    assert result.evidence[0].associations[0].eligible is False


def test_thresholds_and_ambiguity_are_configurable():
    observation = damage(rectangle(0.1, 0.1, 0.3, 0.3), confidence=0.4)
    vehicle_part = part(rectangle(0.2, 0.1, 0.3, 0.3), confidence=0.4)
    assert fused([observation], [vehicle_part]).evidence[0].association_status is AssociationStatus.UNASSIGNED
    policy = VisionFusionPolicy(min_damage_confidence=0.3, min_part_confidence=0.3, min_damage_containment=0.5)
    assert fused([observation], [vehicle_part], policy=policy).evidence[0].association_status is AssociationStatus.ASSIGNED
    assert VisionFusionPolicy.from_toml("configs/vision/vision_fusion_v1.toml") == VisionFusionPolicy()


def test_narrower_ambiguity_margin_can_resolve_close_candidates():
    observation = damage(rectangle(0.1, 0.1, 0.3, 0.3))
    parts = [
        part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
        part(rectangle(0.1, 0.1, 0.265, 0.3), kind=VehiclePartType.HOOD),
    ]
    assert fused([observation], parts).evidence[0].association_status is AssociationStatus.AMBIGUOUS
    evidence = fused([observation], parts, policy=VisionFusionPolicy(ambiguity_margin=0.01)).evidence[0]
    assert evidence.association_status is AssociationStatus.ASSIGNED
    assert evidence.vehicle_part is VehiclePartType.HOOD


@pytest.mark.parametrize("changes", [{"filename": "other.png"}, {"width": 99}, {"height": 99}])
def test_different_images_or_coordinate_spaces_cannot_be_fused(changes):
    with pytest.raises(FusionGeometryError, match="same image"):
        fused([damage()], [part()], **changes)


def test_same_filename_with_different_asset_paths_rejected():
    with pytest.raises(FusionGeometryError, match="different image asset"):
        fuse_image_assessments(
            ImageDamageAssessment(filename="photo.png", asset_reference="first/photo.png", width=10, height=10),
            ImagePartAssessment(filename="photo.png", asset_reference="second/photo.png", width=10, height=10,
                                model_metadata=METADATA, confidence_threshold=0.25),
            damage_model_metadata=METADATA,
        )


@pytest.mark.parametrize("kind,expected", [
    (VehiclePartType.FRONT_BUMPER, DamageType.FRONT_BUMPER),
    (VehiclePartType.HOOD, DamageType.HOOD),
    (VehiclePartType.FRONT_LEFT_LIGHT, DamageType.LEFT_HEADLIGHT),
    (VehiclePartType.FRONT_LIGHT, None),
    (VehiclePartType.FRONT_RIGHT_LIGHT, None),
    (VehiclePartType.FRONT_DOOR, None),
    (VehiclePartType.OTHER, None),
])
def test_business_mapping_is_intentionally_limited(kind, expected):
    assert map_vehicle_part_to_damage(kind) == expected


def test_dataset_object_remains_other_and_cannot_become_a_business_part():
    assert map_vehicle_part_category_name("object") is VehiclePartType.OTHER
    evidence = fused([damage()], [part(kind=VehiclePartType.OTHER)]).evidence[0]
    assert evidence.association_status is AssociationStatus.UNASSIGNED
    assert evidence.associations == []
    assert not hasattr(DamageType, "OBJECT")


def test_assigned_snapshot_cannot_be_tampered_into_a_different_business_part():
    evidence = fused([damage()], [part()]).evidence[0]
    data = evidence.model_dump(round_trip=True)
    data["mapped_damage_type"] = DamageType.LEFT_HEADLIGHT
    with pytest.raises(ValidationError, match="business mapping"):
        type(evidence).model_validate(data)
