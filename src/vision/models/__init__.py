"""Compact, framework-independent observations in EXIF-oriented image space."""

from vision.models._assessments import (
    ClaimVisionAssessment,
    ImageDamageAssessment,
    ImagePartAssessment,
    VehiclePartDetection,
    VisionModelMetadata,
    VisualDamageDetection,
)
from vision.models._association import AssociationMethod, AssociationStatus, DamagePartAssociation
from vision.models._base import NO_DETECTION_SUMMARY, UnitFloat, VisionDomainModel
from vision.models._evidence import VisualDamageEvidence
from vision.models._geometry import NormalizedBoundingBox, NormalizedPoint, NormalizedPolygon
from vision.models._multimodal import ClaimMultimodalVisionAssessment, MultimodalConsistencyAssessment
from vision.models._visual import ImageVisualAssessment

__all__ = [
    "NO_DETECTION_SUMMARY",
    "AssociationMethod",
    "AssociationStatus",
    "ClaimMultimodalVisionAssessment",
    "ClaimVisionAssessment",
    "DamagePartAssociation",
    "ImageDamageAssessment",
    "ImagePartAssessment",
    "ImageVisualAssessment",
    "MultimodalConsistencyAssessment",
    "NormalizedBoundingBox",
    "NormalizedPoint",
    "NormalizedPolygon",
    "UnitFloat",
    "VehiclePartDetection",
    "VisionDomainModel",
    "VisionModelMetadata",
    "VisualDamageDetection",
    "VisualDamageEvidence",
]