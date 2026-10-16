"""Compact, framework-independent observations in EXIF-oriented image space."""

import math
import os
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from claims.enums import DamageType
from vision.fusion_config import VisionFusionPolicy
from vision.taxonomy import VehiclePartType, VisualDamageType, map_vehicle_part_to_damage

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
    """Step29 exterior plus optional holes and disconnected mask components.

    Existing single-contour artifacts retain their loading contract. New mask
    fields preserve observed topology and are checked by the geometry layer.
    """

    points: list[NormalizedPoint] = Field(min_length=3)
    holes: list[list[NormalizedPoint]] = Field(default_factory=list)
    components: list["NormalizedPolygon"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_area(self):
        points = self.points
        area_twice = sum(
            point.x * points[(index + 1) % len(points)].y - points[(index + 1) % len(points)].x * point.y
            for index, point in enumerate(points)
        )
        if abs(area_twice) <= 1e-12:
            raise ValueError("Polygon must enclose a nonzero area")
        for hole in self.holes:
            if len(hole) < 3:
                raise ValueError("Polygon hole must contain at least three points")
            hole_area_twice = sum(
                point.x * hole[(index + 1) % len(hole)].y - hole[(index + 1) % len(hole)].x * point.y
                for index, point in enumerate(hole)
            )
            if abs(hole_area_twice) <= 1e-12:
                raise ValueError("Polygon hole must enclose a nonzero area")
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


class AssociationMethod(str, Enum):
    MASK = "MASK"
    BBOX_FALLBACK = "BBOX_FALLBACK"


class AssociationStatus(str, Enum):
    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"


class DamagePartAssociation(VisionDomainModel):
    damage_detection_index: int = Field(ge=0)
    part_detection_index: int = Field(ge=0)
    candidate_part: VehiclePartDetection
    damage_confidence: UnitFloat
    part_confidence: UnitFloat
    intersection_area: float = Field(ge=0, le=1, allow_inf_nan=False)
    damage_area: float = Field(gt=0, le=1, allow_inf_nan=False)
    part_area: float = Field(gt=0, le=1, allow_inf_nan=False)
    damage_containment_ratio: UnitFloat
    iou: UnitFloat
    association_method: AssociationMethod
    eligible: bool

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.intersection_area > min(self.damage_area, self.part_area) + 1e-12:
            raise ValueError("Intersection area cannot exceed either source area")
        containment = self.intersection_area / self.damage_area
        union = self.damage_area + self.part_area - self.intersection_area
        if union <= 0:
            raise ValueError("Association union area must be positive")
        if not math.isclose(
            self.damage_containment_ratio, containment, rel_tol=1e-9, abs_tol=1e-12
        ) or not math.isclose(self.iou, self.intersection_area / union, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Association containment and IoU must agree with the preserved areas")
        return self


class VisualDamageEvidence(VisionDomainModel):
    """Per-photo observation with the complete reproducible fusion snapshot."""

    filename: str = Field(min_length=1)
    asset_reference: str | None = None
    damage_detection_index: int = Field(ge=0)
    visual_damage_type: VisualDamageType
    damage_confidence: UnitFloat
    vehicle_part: VehiclePartType | None = None
    part_confidence: UnitFloat | None = None
    containment_ratio: UnitFloat | None = None
    mapped_damage_type: DamageType | None = None
    association_status: AssociationStatus
    damage_detection: VisualDamageDetection
    primary_association: DamagePartAssociation | None = None
    associations: list[DamagePartAssociation] = Field(default_factory=list)
    damage_model_metadata: VisionModelMetadata
    part_model_metadata: VisionModelMetadata
    fusion_policy: VisionFusionPolicy

    @model_validator(mode="after")
    def validate_snapshot(self):
        if (self.visual_damage_type, self.damage_confidence) != (
            self.damage_detection.damage_type,
            self.damage_detection.confidence,
        ):
            raise ValueError("Visual evidence must preserve its damage detection")
        for association in self.associations:
            if association.damage_detection_index != self.damage_detection_index:
                raise ValueError("Association must refer to this damage detection")
            if (
                association.damage_confidence != self.damage_confidence
                or association.part_confidence != association.candidate_part.confidence
            ):
                raise ValueError("Association confidences differ from the source detections")
        primary = self.primary_association
        if self.association_status is not AssociationStatus.ASSIGNED:
            if primary is not None or any(
                value is not None
                for value in (
                    self.vehicle_part,
                    self.part_confidence,
                    self.containment_ratio,
                    self.mapped_damage_type,
                )
            ):
                raise ValueError("Unassigned and ambiguous evidence cannot assert a primary vehicle part")
            return self
        if primary is None or primary not in self.associations or not primary.eligible:
            raise ValueError("Assigned evidence requires an eligible preserved primary association")
        if (
            self.vehicle_part != primary.candidate_part.part_type
            or self.part_confidence != primary.part_confidence
            or self.containment_ratio != primary.damage_containment_ratio
            or self.mapped_damage_type != map_vehicle_part_to_damage(primary.candidate_part.part_type)
        ):
            raise ValueError("Primary vehicle part evidence must match the preserved association and business mapping")
        if (
            self.vehicle_part is VehiclePartType.OTHER
            or self.damage_confidence < self.fusion_policy.min_damage_confidence
            or self.part_confidence < self.fusion_policy.min_part_confidence
            or self.containment_ratio < self.fusion_policy.min_damage_containment
        ):
            raise ValueError("Assigned evidence does not meet its fusion policy thresholds")
        return self

    @computed_field
    @property
    def fusion_policy_version(self) -> str:
        return self.fusion_policy.version


class ImageVisualAssessment(VisionDomainModel):
    damage_assessment: ImageDamageAssessment
    part_assessment: ImagePartAssessment
    evidence: list[VisualDamageEvidence]

    @model_validator(mode="after")
    def validate_image_evidence(self):
        damage, parts = self.damage_assessment, self.part_assessment
        if (damage.filename, damage.width, damage.height) != (parts.filename, parts.width, parts.height):
            raise ValueError("Damage and part assessments must describe the same image and oriented dimensions")

        def normalized_reference(reference):
            return os.path.normcase(os.path.abspath(reference))

        known_references = {
            normalized_reference(reference)
            for reference in (damage.asset_reference, parts.asset_reference)
            if reference is not None
        }
        if len(known_references) > 1:
            raise ValueError("Damage and part assessments have different image asset references")
        damage_indices = [item.damage_detection_index for item in self.evidence]
        if len(damage_indices) != len(set(damage_indices)) or set(damage_indices) != set(range(len(damage.detections))):
            raise ValueError("Every source damage detection must have exactly one visual evidence observation")
        for item in self.evidence:
            if item.filename != damage.filename:
                raise ValueError("Visual evidence filename differs from its source image")
            if (
                known_references
                and item.asset_reference is not None
                and normalized_reference(item.asset_reference) not in known_references
            ):
                raise ValueError("Visual evidence asset reference differs from its source image")
            if item.damage_detection != damage.detections[item.damage_detection_index]:
                raise ValueError("Visual evidence damage detection differs from its source detection")
            if item.part_model_metadata != parts.model_metadata:
                raise ValueError("Visual evidence part model metadata differs from its image assessment")
            candidate_indices = [association.part_detection_index for association in item.associations]
            if len(candidate_indices) != len(set(candidate_indices)):
                raise ValueError("Visual evidence cannot duplicate a candidate part detection")
            for association in item.associations:
                if (
                    association.part_detection_index >= len(parts.detections)
                    or association.candidate_part != parts.detections[association.part_detection_index]
                ):
                    raise ValueError("Association candidate part differs from its source part detection")
        # Reproduce deterministic geometry and assignment from stored masks,
        # without invoking a detector. This also protects JSONB round-trips.
        from vision.fusion import validate_image_association_snapshot

        validate_image_association_snapshot(self)
        return self

    @computed_field
    @property
    def filename(self) -> str:
        return self.damage_assessment.filename


class MultimodalConsistencyAssessment(VisionDomainModel):
    declared_damage_types: list[DamageType] = Field(default_factory=list)
    quoted_damage_types: list[DamageType] = Field(default_factory=list)
    visually_supported_declared_damage: list[DamageType] = Field(default_factory=list)
    additional_visual_damage: list[DamageType] = Field(default_factory=list)
    visually_supported_quote_damage: list[DamageType] = Field(default_factory=list)
    visual_damage_unmapped: list[VisualDamageEvidence] = Field(default_factory=list)
    # Informational uncertainty only: absence of a detection is not a risk signal.
    declared_damage_not_visually_confirmed: list[DamageType] = Field(default_factory=list)


class ClaimMultimodalVisionAssessment(VisionDomainModel):
    claim_id: str = Field(min_length=1)
    images: list[ImageVisualAssessment]
    damage_model_metadata: VisionModelMetadata
    part_model_metadata: VisionModelMetadata
    fusion_policy: VisionFusionPolicy
    consistency: MultimodalConsistencyAssessment

    @model_validator(mode="after")
    def validate_claim_snapshot(self):
        for image in self.images:
            if image.part_assessment.model_metadata != self.part_model_metadata:
                raise ValueError("Image part model metadata differs from its claim assessment")
            for detection, metadata in [
                *((item, self.damage_model_metadata) for item in image.damage_assessment.detections),
                *((item, self.part_model_metadata) for item in image.part_assessment.detections),
            ]:
                if (
                    detection.model_identifier is not None
                    and detection.model_identifier != metadata.identifier
                    or detection.model_version is not None
                    and detection.model_version != metadata.version
                ):
                    raise ValueError("Detection model identifier or version differs from its claim assessment")
            for item in image.evidence:
                if (
                    item.damage_model_metadata != self.damage_model_metadata
                    or item.part_model_metadata != self.part_model_metadata
                    or item.fusion_policy != self.fusion_policy
                ):
                    raise ValueError("Visual evidence models or fusion policy differ from its claim assessment")
        declared = set(self.consistency.declared_damage_types)
        quoted = set(self.consistency.quoted_damage_types)
        observations = [item for image in self.images for item in image.evidence]
        mapped = {item.mapped_damage_type for item in observations if item.mapped_damage_type is not None}
        expected = {
            "visually_supported_declared_damage": mapped & declared,
            "additional_visual_damage": mapped - declared,
            "visually_supported_quote_damage": mapped & quoted,
            "declared_damage_not_visually_confirmed": declared - mapped,
        }
        for field, values in expected.items():
            actual = getattr(self.consistency, field)
            if len(actual) != len(set(actual)) or set(actual) != values:
                raise ValueError(f"Multimodal consistency {field} differs from its source observations")
        if self.consistency.visual_damage_unmapped != [
            item for item in observations if item.mapped_damage_type is None
        ]:
            raise ValueError("Unmapped visual damage differs from its source observations")
        return self

    @computed_field
    @property
    def observations(self) -> list[VisualDamageEvidence]:
        # Observations are never deduplicated across photos.
        return [item for image in self.images for item in image.evidence]

    @computed_field
    @property
    def fusion_policy_version(self) -> str:
        return self.fusion_policy.version

    @computed_field
    @property
    def aggregate_mapped_damage_types(self) -> list[DamageType]:
        observed = {item.mapped_damage_type for item in self.observations if item.mapped_damage_type is not None}
        return [damage_type for damage_type in DamageType if damage_type in observed]

    @computed_field
    @property
    def aggregate_vehicle_parts(self) -> list[VehiclePartType]:
        observed = {item.vehicle_part for item in self.observations if item.vehicle_part is not None}
        return [part for part in VehiclePartType if part in observed]

    @computed_field
    @property
    def ambiguous_associations(self) -> list[VisualDamageEvidence]:
        return [item for item in self.observations if item.association_status is AssociationStatus.AMBIGUOUS]
