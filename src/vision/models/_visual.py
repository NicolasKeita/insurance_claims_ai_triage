"""Image-level fusion result that reproduces its evidence from the stored masks."""

import os

from pydantic import computed_field, model_validator

from vision.models._assessments import ImageDamageAssessment, ImagePartAssessment
from vision.models._base import VisionDomainModel
from vision.models._evidence import VisualDamageEvidence


def _normalized_reference(reference: str) -> str:
    return os.path.normcase(os.path.abspath(reference))


def _validated_references(damage: ImageDamageAssessment, parts: ImagePartAssessment) -> set[str]:
    if (damage.filename, damage.width, damage.height) != (parts.filename, parts.width, parts.height):
        raise ValueError("Damage and part assessments must describe the same image and oriented dimensions")
    known_references = {
        _normalized_reference(reference)
        for reference in (damage.asset_reference, parts.asset_reference)
        if reference is not None
    }
    if len(known_references) > 1:
        raise ValueError("Damage and part assessments have different image asset references")
    return known_references


def _require_evidence_coverage(evidence: list[VisualDamageEvidence], damage: ImageDamageAssessment) -> None:
    damage_indices = [item.damage_detection_index for item in evidence]
    if len(damage_indices) != len(set(damage_indices)) or set(damage_indices) != set(range(len(damage.detections))):
        raise ValueError("Every source damage detection must have exactly one visual evidence observation")


def _validate_evidence_row(
    item: VisualDamageEvidence,
    damage: ImageDamageAssessment,
    parts: ImagePartAssessment,
    known_references: set[str],
) -> None:
    if item.filename != damage.filename:
        raise ValueError("Visual evidence filename differs from its source image")
    if (
        known_references
        and item.asset_reference is not None
        and _normalized_reference(item.asset_reference) not in known_references
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


class ImageVisualAssessment(VisionDomainModel):
    damage_assessment: ImageDamageAssessment
    part_assessment: ImagePartAssessment
    evidence: list[VisualDamageEvidence]

    @model_validator(mode="after")
    def validate_image_evidence(self):
        """Reproduce deterministic geometry and assignment from stored masks, without invoking a detector.

        This also protects JSONB round-trips.
        """
        damage, parts = self.damage_assessment, self.part_assessment
        known_references = _validated_references(damage, parts)
        _require_evidence_coverage(self.evidence, damage)
        for item in self.evidence:
            _validate_evidence_row(item, damage, parts, known_references)
        from vision.fusion import validate_image_association_snapshot

        validate_image_association_snapshot(self)
        return self

    @computed_field
    @property
    def filename(self) -> str:
        return self.damage_assessment.filename