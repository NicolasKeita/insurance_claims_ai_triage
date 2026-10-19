"""Fuse one photo's damage and part observations into preserved evidence rows."""

from vision.fusion._association import (
    damage_part_candidates,
    select_primary_association,
    validate_same_image,
)
from vision.fusion._overlap import require_valid_part_polygons
from vision.fusion_config import DEFAULT_FUSION_POLICY, VisionFusionPolicy
from vision.models import (
    ImageDamageAssessment,
    ImagePartAssessment,
    ImageVisualAssessment,
    VisionModelMetadata,
    VisualDamageEvidence,
)
from vision.taxonomy import map_vehicle_part_to_damage


def _image_evidence(
    damage_assessment: ImageDamageAssessment,
    part_assessment: ImagePartAssessment,
    damage_model_metadata: VisionModelMetadata,
    policy: VisionFusionPolicy,
) -> list[VisualDamageEvidence]:
    """One evidence row per damage detection, keeping every ranked part candidate."""
    evidence = []
    for damage_index, damage in enumerate(damage_assessment.detections):
        candidates = damage_part_candidates(damage, damage_index, part_assessment.detections, policy)
        status, primary = select_primary_association(candidates, policy)
        evidence.append(
            VisualDamageEvidence(
                filename=damage_assessment.filename,
                asset_reference=damage_assessment.asset_reference or part_assessment.asset_reference,
                damage_detection_index=damage_index,
                visual_damage_type=damage.damage_type,
                damage_confidence=damage.confidence,
                vehicle_part=primary.candidate_part.part_type if primary else None,
                part_confidence=primary.part_confidence if primary else None,
                containment_ratio=primary.damage_containment_ratio if primary else None,
                mapped_damage_type=map_vehicle_part_to_damage(primary.candidate_part.part_type) if primary else None,
                association_status=status,
                damage_detection=damage,
                primary_association=primary,
                associations=candidates,
                damage_model_metadata=damage_model_metadata,
                part_model_metadata=part_assessment.model_metadata,
                fusion_policy=policy,
            )
        )
    return evidence


def fuse_image_assessments(
    damage_assessment: ImageDamageAssessment,
    part_assessment: ImagePartAssessment,
    *,
    damage_model_metadata: VisionModelMetadata,
    policy: VisionFusionPolicy | None = None,
) -> ImageVisualAssessment:
    """Fuse one photo without repairing geometry or inventing part assignments."""
    validate_same_image(damage_assessment, part_assessment)
    policy = policy or DEFAULT_FUSION_POLICY
    require_valid_part_polygons(part_assessment)
    return ImageVisualAssessment(
        damage_assessment=damage_assessment,
        part_assessment=part_assessment,
        evidence=_image_evidence(damage_assessment, part_assessment, damage_model_metadata, policy),
    )
