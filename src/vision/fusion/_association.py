"""Damage-to-part candidate ranking and primary association selection."""

import os

from vision.fusion._overlap import (
    FusionGeometryError,
    calculate_overlap,
    validated_polygon,
)
from vision.fusion_config import VisionFusionPolicy
from vision.models import (
    AssociationStatus,
    DamagePartAssociation,
    ImageDamageAssessment,
    ImagePartAssessment,
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
    indexed_part: tuple[int, VehiclePartDetection],
    overlap,
    policy: VisionFusionPolicy,
) -> DamagePartAssociation:
    part_index, part = indexed_part
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
        candidates.append(_association(damage, damage_index, (part_index, part), overlap, policy))
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
