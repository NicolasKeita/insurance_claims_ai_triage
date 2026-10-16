"""Damage-to-part candidate ranking and primary association selection."""

import math
import os

from vision.fusion._overlap import (
    FusionGeometryError,
    calculate_overlap,
    require_valid_part_polygons,
    validated_polygon,
)
from vision.fusion_config import VisionFusionPolicy
from vision.models import (
    AssociationStatus,
    DamagePartAssociation,
    ImageDamageAssessment,
    ImagePartAssessment,
    ImageVisualAssessment,
    VehiclePartDetection,
    VisualDamageDetection,
)
from vision.taxonomy import VehiclePartType


def validate_same_image(damage: ImageDamageAssessment, parts: ImagePartAssessment) -> None:
    if (damage.filename, damage.width, damage.height) != (parts.filename, parts.width, parts.height):
        raise FusionGeometryError("Damage and part assessments must refer to the same image and oriented dimensions")
    if damage.asset_reference is not None and parts.asset_reference is not None:
        damage_reference = os.path.normcase(os.path.abspath(damage.asset_reference))
        part_reference = os.path.normcase(os.path.abspath(parts.asset_reference))
        if damage_reference != part_reference:
            raise FusionGeometryError("Damage and part assessments have different image asset references")


def _association(
    damage: VisualDamageDetection,
    damage_index: int,
    part_index: int,
    part: VehiclePartDetection,
    overlap,
    policy: VisionFusionPolicy,
) -> DamagePartAssociation:
    return DamagePartAssociation(
        damage_detection_index=damage_index,
        part_detection_index=part_index,
        candidate_part=part,
        damage_confidence=damage.confidence,
        part_confidence=part.confidence,
        intersection_area=overlap.intersection_area,
        damage_area=overlap.damage_area,
        part_area=overlap.part_area,
        damage_containment_ratio=overlap.damage_containment_ratio,
        iou=overlap.iou,
        association_method=overlap.association_method,
        eligible=(
            damage.confidence >= policy.min_damage_confidence
            and part.confidence >= policy.min_part_confidence
            and overlap.damage_containment_ratio >= policy.min_damage_containment
        ),
    )


def damage_part_candidates(
    damage: VisualDamageDetection,
    damage_index: int,
    parts: list[VehiclePartDetection],
    policy: VisionFusionPolicy,
) -> list[DamagePartAssociation]:
    """Preserve every positive-overlap business candidate in score order."""
    if damage.polygon is not None:
        validated_polygon(damage.polygon)
    candidates = []
    for part_index, part in enumerate(parts):
        if part.part_type is VehiclePartType.OTHER:
            continue
        overlap = calculate_overlap(damage, part)
        if overlap.intersection_area == 0:
            continue
        candidates.append(_association(damage, damage_index, part_index, part, overlap, policy))
    return sorted(candidates, key=lambda item: (-item.damage_containment_ratio, item.part_detection_index))


def select_primary_association(
    candidates: list[DamagePartAssociation],
    policy: VisionFusionPolicy,
) -> tuple[AssociationStatus, DamagePartAssociation | None]:
    eligible = sorted(
        (candidate for candidate in candidates if candidate.eligible),
        key=lambda item: (-item.damage_containment_ratio, item.part_detection_index),
    )
    if not eligible:
        return AssociationStatus.UNASSIGNED, None
    if (
        len(eligible) > 1
        and round(eligible[0].damage_containment_ratio - eligible[1].damage_containment_ratio, 15)
        <= policy.ambiguity_margin
    ):
        return AssociationStatus.AMBIGUOUS, None
    return AssociationStatus.ASSIGNED, eligible[0]


def _candidates_match(candidate: DamagePartAssociation, observed: DamagePartAssociation) -> None:
    if candidate.association_method is not observed.association_method or candidate.eligible != observed.eligible:
        raise ValueError("Association method or eligibility differs from its masks and fusion policy")
    for metric in ("intersection_area", "damage_area", "part_area", "damage_containment_ratio", "iou"):
        if not math.isclose(getattr(candidate, metric), getattr(observed, metric), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Association {metric} differs from the preserved source masks")


def validate_image_association_snapshot(image: ImageVisualAssessment) -> None:
    """Check persisted metrics/candidates against the original observed masks."""
    require_valid_part_polygons(image.part_assessment)
    for item in image.evidence:
        expected = damage_part_candidates(
            item.damage_detection,
            item.damage_detection_index,
            image.part_assessment.detections,
            item.fusion_policy,
        )
        expected_by_index = {candidate.part_detection_index: candidate for candidate in expected}
        actual_by_index = {candidate.part_detection_index: candidate for candidate in item.associations}
        if actual_by_index.keys() != expected_by_index.keys():
            raise ValueError("Preserved association candidates differ from positive overlaps of the source masks")
        for index, candidate in actual_by_index.items():
            _candidates_match(candidate, expected_by_index[index])
        status, primary = select_primary_association(expected, item.fusion_policy)
        primary_index = item.primary_association.part_detection_index if item.primary_association is not None else None
        if item.association_status is not status or primary_index != (
            primary.part_detection_index if primary is not None else None
        ):
            raise ValueError("Association status or primary part differs from containment ranking and ambiguity policy")