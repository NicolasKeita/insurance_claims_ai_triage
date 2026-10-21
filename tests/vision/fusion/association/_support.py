"""Support."""

from tests.vision.fusion.association._constants import METADATA
from vision.fusion import fuse_image_assessments
from vision.models import (
    ImageDamageAssessment,
    ImagePartAssessment,
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
    VehiclePartDetection,
    VisualDamageDetection,
)
from vision.taxonomy import (
    VehiclePartType,
    VisualDamageType,
)


def rectangle(x_min, y_min, x_max, y_max):
    return NormalizedBoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def polygon_for(bounds):
    return NormalizedPolygon(
        points=[
            NormalizedPoint(x=bounds.x_min, y=bounds.y_min),
            NormalizedPoint(x=bounds.x_max, y=bounds.y_min),
            NormalizedPoint(x=bounds.x_max, y=bounds.y_max),
            NormalizedPoint(x=bounds.x_min, y=bounds.y_max),
        ]
    )


def damage(bounds=None, *, confidence=0.9, mask=True):
    bounds = bounds or rectangle(0.1, 0.1, 0.2, 0.2)
    return VisualDamageDetection(
        damage_type=VisualDamageType.SCRATCH,
        confidence=confidence,
        bounding_box=bounds,
        polygon=polygon_for(bounds) if mask else None,
    )


def part(bounds=None, *, kind=VehiclePartType.HOOD, confidence=0.9, mask=True):
    bounds = bounds or rectangle(0, 0, 1, 1)
    return VehiclePartDetection(
        part_type=kind,
        confidence=confidence,
        bounding_box=bounds,
        polygon=polygon_for(bounds) if mask else None,
    )


def fused(detections, parts, *, policy=None, **part_image_changes):
    return fuse_image_assessments(
        ImageDamageAssessment(filename="synthetic.png", width=100, height=100, detections=detections),
        ImagePartAssessment(
            **{
                "filename": "synthetic.png",
                "width": 100,
                "height": 100,
                "detections": parts,
                "model_metadata": METADATA,
                "confidence_threshold": 0.25,
            }
            | part_image_changes
        ),
        damage_model_metadata=METADATA,
        policy=policy,
    )
