"""Compact, framework-independent observations in EXIF-oriented image space."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from vision.taxonomy import VisualDamageType

UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NO_DETECTION_SUMMARY = "No supported damage detected above the configured confidence threshold."


class VisionDomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalizedBoundingBox(VisionDomainModel):
    x_min: UnitFloat
    y_min: UnitFloat
    x_max: UnitFloat
    y_max: UnitFloat

    @model_validator(mode="after")
    def validate_extent(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box must have positive width and height")
        return self


class NormalizedPoint(VisionDomainModel):
    x: UnitFloat
    y: UnitFloat


class NormalizedPolygon(VisionDomainModel):
    points: list[NormalizedPoint] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_area(self):
        points = self.points
        area_twice = sum(
            point.x * points[(index + 1) % len(points)].y
            - points[(index + 1) % len(points)].x * point.y
            for index, point in enumerate(points)
        )
        if abs(area_twice) <= 1e-12:
            raise ValueError("Polygon must enclose a nonzero area")
        return self


class VisualDamageDetection(VisionDomainModel):
    damage_type: VisualDamageType
    confidence: UnitFloat
    bounding_box: NormalizedBoundingBox
    polygon: NormalizedPolygon | None = None
    model_identifier: str | None = None
    model_version: str | None = None


class VisionModelMetadata(VisionDomainModel):
    identifier: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    task: Literal["instance_segmentation"] = "instance_segmentation"
    version: str | None = None
    framework_version: str | None = None


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
