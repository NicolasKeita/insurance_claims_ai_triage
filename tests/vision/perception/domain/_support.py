"""Support."""

from vision.models import (
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
    VisualDamageDetection,
)
from vision.taxonomy import VisualDamageType


def detection(kind=VisualDamageType.DENT, confidence=0.8):
    return VisualDamageDetection(
        damage_type=kind,
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9),
        polygon=NormalizedPolygon(
            points=[NormalizedPoint(x=0.1, y=0.2), NormalizedPoint(x=0.8, y=0.2), NormalizedPoint(x=0.8, y=0.9)]
        ),
    )
