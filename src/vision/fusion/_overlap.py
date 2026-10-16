"""Mask-first same-image geometry; containment, not IoU, decides association."""

import math
from dataclasses import dataclass

from pydantic import ValidationError

from vision.geometry import PolygonGeometryError, validated_polygon_geometry
from vision.models import (
    AssociationMethod,
    ImagePartAssessment,
    NormalizedBoundingBox,
    NormalizedPolygon,
    VehiclePartDetection,
    VisualDamageDetection,
)
from vision.taxonomy import VehiclePartType


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


def validated_polygon(polygon: NormalizedPolygon):
    try:
        return validated_polygon_geometry(polygon)
    except PolygonGeometryError as error:
        raise FusionGeometryError("Invalid normalized polygon; bounding-box fallback is forbidden") from error


def validated_box(bounding_box: NormalizedBoundingBox):
    from shapely.geometry import box

    try:
        validated = NormalizedBoundingBox.model_validate(bounding_box.model_dump())
    except ValidationError as error:
        raise FusionGeometryError("Invalid normalized bounding box") from error
    return box(validated.x_min, validated.y_min, validated.x_max, validated.y_max)


def require_valid_part_polygons(parts: ImagePartAssessment) -> None:
    """Every non-OTHER part polygon must be usable geometry, never a silent fallback."""
    for part in parts.detections:
        if part.part_type is not VehiclePartType.OTHER and part.polygon is not None:
            validated_polygon(part.polygon)


def _overlap_geometry(damage: VisualDamageDetection, part: VehiclePartDetection) -> tuple:
    damage_polygon = validated_polygon(damage.polygon) if damage.polygon is not None else None
    part_polygon = validated_polygon(part.polygon) if part.polygon is not None else None
    if damage_polygon is not None and part_polygon is not None:
        return damage_polygon, part_polygon, AssociationMethod.MASK
    return validated_box(damage.bounding_box), validated_box(part.bounding_box), AssociationMethod.BBOX_FALLBACK


def calculate_overlap(damage: VisualDamageDetection, part: VehiclePartDetection) -> GeometryOverlap:
    """All areas refer to normalized, EXIF-oriented coordinates of one image.

    Invalid supplied polygons fail explicitly, including when the other mask is
    missing. Only unavailable/unusable contours already surfaced as ``None`` by
    an adapter can use BBOX_FALLBACK, which is kept in the evidence snapshot.

    Bound roundoff at polygon edges; geometry is never repaired or replaced.
    """
    from shapely.errors import GEOSException

    damage_geometry, part_geometry, method = _overlap_geometry(damage, part)
    try:
        damage_area, part_area = float(damage_geometry.area), float(part_geometry.area)
        if not all(math.isfinite(area) and area > 0 for area in (damage_area, part_area)):
            raise FusionGeometryError("Damage and part geometry must have finite positive area")
        intersection = float(damage_geometry.intersection(part_geometry).area)
    except GEOSException as error:
        raise FusionGeometryError("GEOS could not intersect valid damage and vehicle part geometry") from error
    intersection = max(0.0, min(intersection, damage_area, part_area))
    union = damage_area + part_area - intersection
    return GeometryOverlap(
        intersection_area=intersection,
        damage_area=damage_area,
        part_area=part_area,
        damage_containment_ratio=min(1.0, round(intersection / damage_area, 15)),
        iou=min(1.0, intersection / union),
        association_method=method,
    )