"""Vehicle damage perception; imports never load a model or download weights."""

from vision.models import (
    ClaimVisionAssessment,
    ImageDamageAssessment,
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
    VisualDamageDetection,
    VisionModelMetadata,
    AssociationMethod,
    AssociationStatus,
    ClaimMultimodalVisionAssessment,
    DamagePartAssociation,
    ImagePartAssessment,
    ImageVisualAssessment,
    MultimodalConsistencyAssessment,
    VehiclePartDetection,
    VisualDamageEvidence,
)
from vision.taxonomy import VehiclePartType, VisualDamageType

__all__ = [
    "ClaimVisionAssessment", "ImageDamageAssessment", "NormalizedBoundingBox",
    "NormalizedPoint", "NormalizedPolygon", "VisualDamageDetection",
    "VisualDamageType", "VisionModelMetadata",
    "AssociationMethod", "AssociationStatus", "ClaimMultimodalVisionAssessment",
    "DamagePartAssociation", "ImagePartAssessment", "ImageVisualAssessment",
    "MultimodalConsistencyAssessment", "VehiclePartDetection", "VehiclePartType",
    "VisualDamageEvidence",
]
