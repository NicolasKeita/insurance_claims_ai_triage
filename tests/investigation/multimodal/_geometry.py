"""Geometry."""

from vision.models import (
    DamagePartAssociation,
    NormalizedBoundingBox,
    NormalizedPolygon,
    VehiclePartDetection,
    VisualDamageDetection,
)
from vision.taxonomy import VehiclePartType, VisualDamageType


def polygon_factory():
    def rectangle(box):
        return NormalizedPolygon(
            points=[
                {"x": box.x_min, "y": box.y_min},
                {"x": box.x_max, "y": box.y_min},
                {"x": box.x_max, "y": box.y_max},
                {"x": box.x_min, "y": box.y_max},
            ]
        )

    return rectangle


def visual_geometry():
    damage_box = NormalizedBoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    part_box = NormalizedBoundingBox(x_min=0, y_min=0, x_max=0.5, y_max=0.5)

    rectangle = polygon_factory()
    damage = VisualDamageDetection(
        damage_type=VisualDamageType.SCRATCH,
        confidence=0.87,
        bounding_box=damage_box,
        polygon=rectangle(damage_box),
        model_identifier="fixture-damage",
        model_version="damage-v1",
    )
    part = VehiclePartDetection(
        part_type=VehiclePartType.HOOD,
        confidence=0.94,
        bounding_box=part_box,
        polygon=rectangle(part_box),
        model_identifier="fixture-parts",
        model_version="parts-v1",
    )
    association = visual_association(part)
    return damage, part, association


def visual_association(part):
    return DamagePartAssociation(
        damage_detection_index=0,
        part_detection_index=0,
        candidate_part=part,
        damage_confidence=0.87,
        part_confidence=0.94,
        intersection_area=0.01,
        damage_area=0.01,
        part_area=0.25,
        damage_containment_ratio=1,
        iou=0.04,
        association_method="MASK",
        eligible=True,
    )
