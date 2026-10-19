"""Per-image damage and vehicle-part assessments with their model identity."""

from typing import Literal

from pydantic import Field, computed_field

from vision.models._base import NO_DETECTION_SUMMARY, UnitFloat, VisionDomainModel
from vision.models._geometry import NormalizedBoundingBox, NormalizedPolygon
from vision.taxonomy import VehiclePartType, VisualDamageType


class VisionModelMetadata(VisionDomainModel):
    identifier: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    task: Literal["instance_segmentation"] = "instance_segmentation"
    version: str | None = None
    framework_version: str | None = None


class VisualDamageDetection(VisionDomainModel):
    damage_type: VisualDamageType
    confidence: UnitFloat
    bounding_box: NormalizedBoundingBox
    polygon: NormalizedPolygon | None = None
    model_identifier: str | None = None
    model_version: str | None = None


class ImageDamageAssessment(VisionDomainModel):
    filename: str = Field(min_length=1)
    asset_reference: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[VisualDamageDetection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @computed_field
    @property
    def observation_summary(self) -> str:
        return "Supported damage observations detected." if self.detections else NO_DETECTION_SUMMARY


class ClaimVisionAssessment(VisionDomainModel):
    claim_id: str = Field(min_length=1)
    images: list[ImageDamageAssessment]
    model_metadata: VisionModelMetadata
    confidence_threshold: UnitFloat

    @computed_field
    @property
    def aggregate_damage_types(self) -> list[VisualDamageType]:
        observed = {item.damage_type for image in self.images for item in image.detections}
        return [damage_type for damage_type in VisualDamageType if damage_type in observed]


class VehiclePartDetection(VisionDomainModel):
    part_type: VehiclePartType
    confidence: UnitFloat
    bounding_box: NormalizedBoundingBox
    polygon: NormalizedPolygon | None = None
    model_identifier: str | None = None
    model_version: str | None = None


class ImagePartAssessment(VisionDomainModel):
    filename: str = Field(min_length=1)
    asset_reference: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[VehiclePartDetection] = Field(default_factory=list)
    model_metadata: VisionModelMetadata
    confidence_threshold: UnitFloat
    warnings: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def detection_count(self) -> int:
        return len(self.detections)
