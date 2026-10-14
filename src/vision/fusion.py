"""Mask-first same-image geometry; containment, not IoU, decides association."""

import math
import os
from dataclasses import dataclass

from pydantic import ValidationError

from vision.fusion_config import DEFAULT_FUSION_POLICY, VisionFusionPolicy
from vision.geometry import PolygonGeometryError, validated_polygon_geometry
from vision.models import (
    AssociationMethod, AssociationStatus, DamagePartAssociation,
    ImageDamageAssessment, ImagePartAssessment, ImageVisualAssessment,
    NormalizedBoundingBox, NormalizedPolygon, VehiclePartDetection,
    VisualDamageDetection, VisualDamageEvidence, VisionModelMetadata,
)
from vision.taxonomy import VehiclePartType, map_vehicle_part_to_damage


class FusionGeometryError(ValueError):
    pass


@dataclass(frozen=True)
class GeometryOverlap:
    intersection_area: float
    damage_area: float
    part_area: float
    damage_containment_ratio: float
    iou: float
    association_method: AssociationMethod


def _validated_polygon(polygon: NormalizedPolygon):
    try:
        return validated_polygon_geometry(polygon)
    except PolygonGeometryError as error:
        raise FusionGeometryError("Invalid normalized polygon; bounding-box fallback is forbidden") from error


def _validated_box(bounding_box: NormalizedBoundingBox):
    from shapely.geometry import box

    try:
        validated = NormalizedBoundingBox.model_validate(bounding_box.model_dump())
    except ValidationError as error:
        raise FusionGeometryError("Invalid normalized bounding box") from error
    return box(validated.x_min, validated.y_min, validated.x_max, validated.y_max)


def calculate_overlap(damage: VisualDamageDetection, part: VehiclePartDetection) -> GeometryOverlap:
    """All areas refer to normalized, EXIF-oriented coordinates of one image.

    Invalid supplied polygons fail explicitly, including when the other mask is
    missing. Only unavailable/unusable contours already surfaced as ``None`` by
    an adapter can use BBOX_FALLBACK, which is kept in the evidence snapshot.
    """
    from shapely.errors import GEOSException

    damage_polygon = _validated_polygon(damage.polygon) if damage.polygon is not None else None
    part_polygon = _validated_polygon(part.polygon) if part.polygon is not None else None
    if damage_polygon is not None and part_polygon is not None:
        damage_geometry, part_geometry = damage_polygon, part_polygon
        method = AssociationMethod.MASK
    else:
        damage_geometry, part_geometry = _validated_box(damage.bounding_box), _validated_box(part.bounding_box)
        method = AssociationMethod.BBOX_FALLBACK
    try:
        damage_area, part_area = float(damage_geometry.area), float(part_geometry.area)
        if not all(math.isfinite(area) and area > 0 for area in (damage_area, part_area)):
            raise FusionGeometryError("Damage and part geometry must have finite positive area")
        intersection = float(damage_geometry.intersection(part_geometry).area)
    except GEOSException as error:
        raise FusionGeometryError("GEOS could not intersect valid damage and vehicle part geometry") from error
    # Bound roundoff at polygon edges; geometry is never repaired or replaced.
    intersection = max(0.0, min(intersection, damage_area, part_area))
    union = damage_area + part_area - intersection
    return GeometryOverlap(
        intersection_area=intersection, damage_area=damage_area, part_area=part_area,
        damage_containment_ratio=min(1.0, round(intersection / damage_area, 15)),
        iou=min(1.0, intersection / union), association_method=method,
    )


def _validate_same_image(damage: ImageDamageAssessment, parts: ImagePartAssessment):
    if (damage.filename, damage.width, damage.height) != (parts.filename, parts.width, parts.height):
        raise FusionGeometryError("Damage and part assessments must refer to the same image and oriented dimensions")
    if damage.asset_reference is not None and parts.asset_reference is not None:
        damage_reference = os.path.normcase(os.path.abspath(damage.asset_reference))
        part_reference = os.path.normcase(os.path.abspath(parts.asset_reference))
        if damage_reference != part_reference:
            raise FusionGeometryError("Damage and part assessments have different image asset references")


def fuse_image_assessments(
    damage_assessment: ImageDamageAssessment,
    part_assessment: ImagePartAssessment,
    *, damage_model_metadata: VisionModelMetadata,
    policy: VisionFusionPolicy | None = None,
) -> ImageVisualAssessment:
    _validate_same_image(damage_assessment, part_assessment)
    policy = policy or DEFAULT_FUSION_POLICY
    evidence = []
    for part in part_assessment.detections:
        if part.part_type is not VehiclePartType.OTHER and part.polygon is not None:
            _validated_polygon(part.polygon)
    # Validate every supplied damage contour even if there are no part detections.
    for damage_index, damage in enumerate(damage_assessment.detections):
        if damage.polygon is not None:
            _validated_polygon(damage.polygon)
        candidates = []
        for part_index, part in enumerate(part_assessment.detections):
            if part.part_type is VehiclePartType.OTHER:
                continue
            overlap = calculate_overlap(damage, part)
            if overlap.intersection_area == 0:
                continue
            candidates.append(DamagePartAssociation(
                damage_detection_index=damage_index, part_detection_index=part_index,
                candidate_part=part, damage_confidence=damage.confidence,
                part_confidence=part.confidence,
                intersection_area=overlap.intersection_area,
                damage_area=overlap.damage_area, part_area=overlap.part_area,
                damage_containment_ratio=overlap.damage_containment_ratio, iou=overlap.iou,
                association_method=overlap.association_method,
                eligible=(
                    damage.confidence >= policy.min_damage_confidence
                    and part.confidence >= policy.min_part_confidence
                    and overlap.damage_containment_ratio >= policy.min_damage_containment
                ),
            ))
        candidates.sort(key=lambda item: (-item.damage_containment_ratio, item.part_detection_index))
        eligible = [candidate for candidate in candidates if candidate.eligible]
        primary = None
        status = AssociationStatus.UNASSIGNED
        if eligible:
            if (len(eligible) > 1 and
                    round(eligible[0].damage_containment_ratio - eligible[1].damage_containment_ratio, 15) <= policy.ambiguity_margin):
                status = AssociationStatus.AMBIGUOUS
            else:
                primary = eligible[0]
                status = AssociationStatus.ASSIGNED
        evidence.append(VisualDamageEvidence(
            filename=damage_assessment.filename,
            asset_reference=damage_assessment.asset_reference or part_assessment.asset_reference,
            damage_detection_index=damage_index,
            visual_damage_type=damage.damage_type, damage_confidence=damage.confidence,
            vehicle_part=primary.candidate_part.part_type if primary else None,
            part_confidence=primary.part_confidence if primary else None,
            containment_ratio=primary.damage_containment_ratio if primary else None,
            mapped_damage_type=map_vehicle_part_to_damage(primary.candidate_part.part_type) if primary else None,
            association_status=status, damage_detection=damage,
            primary_association=primary, associations=candidates,
            damage_model_metadata=damage_model_metadata,
            part_model_metadata=part_assessment.model_metadata, fusion_policy=policy,
        ))
    return ImageVisualAssessment(
        damage_assessment=damage_assessment, part_assessment=part_assessment, evidence=evidence,
    )
