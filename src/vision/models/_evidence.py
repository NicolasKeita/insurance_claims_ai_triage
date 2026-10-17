"""Per-photo evidence rows with their complete reproducible fusion snapshot."""

from pydantic import Field, computed_field, model_validator

from claims.enums import DamageType
from vision.fusion_config import VisionFusionPolicy
from vision.models._assessments import VisualDamageDetection, VisionModelMetadata
from vision.models._association import AssociationStatus, DamagePartAssociation
from vision.models._base import UnitFloat, VisionDomainModel
from vision.taxonomy import VehiclePartType, VisualDamageType, map_vehicle_part_to_damage


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
            self._validate_association(association)
        if self.association_status is AssociationStatus.ASSIGNED:
            self._validate_primary()
        else:
            self._require_no_primary()
        return self

    def _validate_association(self, association: DamagePartAssociation) -> None:
        if association.damage_detection_index != self.damage_detection_index:
            raise ValueError("Association must refer to this damage detection")
        if (
            association.damage_confidence != self.damage_confidence
            or association.part_confidence != association.candidate_part.confidence
        ):
            raise ValueError("Association confidences differ from the source detections")

    def _require_no_primary(self) -> None:
        if self.primary_association is not None or any(
            value is not None
            for value in (self.vehicle_part, self.part_confidence, self.containment_ratio, self.mapped_damage_type)
        ):
            raise ValueError("Unassigned and ambiguous evidence cannot assert a primary vehicle part")

    def _validate_primary(self) -> None:
        primary = self.primary_association
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

    @computed_field
    @property
    def fusion_policy_version(self) -> str:
        return self.fusion_policy.version