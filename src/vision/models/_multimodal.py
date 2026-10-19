"""Claim-level multimodal snapshot and its declared/quoted/observed consistency."""

from pydantic import Field, computed_field, model_validator

from claims.enums import DamageType
from vision.fusion_config import VisionFusionPolicy
from vision.models._assessments import VisionModelMetadata
from vision.models._association import AssociationStatus
from vision.models._base import VisionDomainModel
from vision.models._evidence import VisualDamageEvidence
from vision.models._visual import ImageVisualAssessment
from vision.taxonomy import VehiclePartType


class MultimodalConsistencyAssessment(VisionDomainModel):
    declared_damage_types: list[DamageType] = Field(default_factory=list)
    quoted_damage_types: list[DamageType] = Field(default_factory=list)
    visually_supported_declared_damage: list[DamageType] = Field(default_factory=list)
    additional_visual_damage: list[DamageType] = Field(default_factory=list)
    visually_supported_quote_damage: list[DamageType] = Field(default_factory=list)
    visual_damage_unmapped: list[VisualDamageEvidence] = Field(default_factory=list)
    # Informational uncertainty only: absence of a detection is not a risk signal.
    declared_damage_not_visually_confirmed: list[DamageType] = Field(default_factory=list)


def _validate_image_metadata(
    image: ImageVisualAssessment,
    damage_metadata: VisionModelMetadata,
    part_metadata: VisionModelMetadata,
    policy: VisionFusionPolicy,
) -> None:
    if image.part_assessment.model_metadata != part_metadata:
        raise ValueError("Image part model metadata differs from its claim assessment")
    for detection, metadata in [
        *((item, damage_metadata) for item in image.damage_assessment.detections),
        *((item, part_metadata) for item in image.part_assessment.detections),
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
            item.damage_model_metadata != damage_metadata
            or item.part_model_metadata != part_metadata
            or item.fusion_policy != policy
        ):
            raise ValueError("Visual evidence models or fusion policy differ from its claim assessment")


def _validate_consistency(consistency: MultimodalConsistencyAssessment, images: list[ImageVisualAssessment]) -> None:
    declared = set(consistency.declared_damage_types)
    quoted = set(consistency.quoted_damage_types)
    observations = [item for image in images for item in image.evidence]
    mapped = {item.mapped_damage_type for item in observations if item.mapped_damage_type is not None}
    expected = {
        "visually_supported_declared_damage": mapped & declared,
        "additional_visual_damage": mapped - declared,
        "visually_supported_quote_damage": mapped & quoted,
        "declared_damage_not_visually_confirmed": declared - mapped,
    }
    for field, values in expected.items():
        actual = getattr(consistency, field)
        if len(actual) != len(set(actual)) or set(actual) != values:
            raise ValueError(f"Multimodal consistency {field} differs from its source observations")
    if consistency.visual_damage_unmapped != [item for item in observations if item.mapped_damage_type is None]:
        raise ValueError("Unmapped visual damage differs from its source observations")


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
            _validate_image_metadata(image, self.damage_model_metadata, self.part_model_metadata, self.fusion_policy)
        _validate_consistency(self.consistency, self.images)
        return self

    @computed_field
    @property
    def observations(self) -> list[VisualDamageEvidence]:
        """Observations are never deduplicated across photos."""
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
