"""Vehicle damage perception; imports never load a model or download weights."""

from vision.models import (
    ClaimVisionAssessment,
    ImageDamageAssessment,
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
    VisualDamageDetection,
    VisionModelMetadata,
)
from vision.taxonomy import VisualDamageType

__all__ = [
    "ClaimVisionAssessment", "ImageDamageAssessment", "NormalizedBoundingBox",
    "NormalizedPoint", "NormalizedPolygon", "VisualDamageDetection",
    "VisualDamageType", "VisionModelMetadata",
]
