"""Vehicle damage perception; imports never load a model or download weights."""

from vision.models import (
    AssociationMethod,
    AssociationStatus,
    ClaimMultimodalVisionAssessment,
    ClaimVisionAssessment,
    DamagePartAssociation,
    ImageDamageAssessment,
    ImagePartAssessment,
    ImageVisualAssessment,
    MultimodalConsistencyAssessment,
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
    VehiclePartDetection,
    VisionModelMetadata,
    VisualDamageDetection,
    VisualDamageEvidence,
)
from vision.taxonomy import VehiclePartType, VisualDamageType

__all__ = [
    "ClaimVisionAssessment",
    "ImageDamageAssessment",
    "NormalizedBoundingBox",
    "NormalizedPoint",
    "NormalizedPolygon",
    "VisualDamageDetection",
    "VisualDamageType",
    "VisionModelMetadata",
    "AssociationMethod",
    "AssociationStatus",
    "ClaimMultimodalVisionAssessment",
    "DamagePartAssociation",
    "ImagePartAssessment",
    "ImageVisualAssessment",
    "MultimodalConsistencyAssessment",
    "VehiclePartDetection",
    "VehiclePartType",
    "VisualDamageEvidence",
]
