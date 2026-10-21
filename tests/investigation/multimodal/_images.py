"""Images."""

from dataclasses import dataclass

from tests.investigation.multimodal._geometry import visual_geometry
from vision.fusion_config import DEFAULT_FUSION_POLICY
from vision.models import (
    DamagePartAssociation,
    ImageDamageAssessment,
    ImagePartAssessment,
    ImageVisualAssessment,
    VehiclePartDetection,
    VisionModelMetadata,
    VisualDamageDetection,
    VisualDamageEvidence,
)
from vision.taxonomy import VehiclePartType, VisualDamageType


@dataclass
class VisualFixture:
    """A consistent per-photo model snapshot shared by visual evidence builders."""

    damage_model: VisionModelMetadata
    part_model: VisionModelMetadata
    damage: VisualDamageDetection
    part: VehiclePartDetection
    association: DamagePartAssociation


def visual_images(image_count, include_detection, damage_model, part_model):
    context = VisualFixture(damage_model, part_model, *visual_geometry())
    images = []
    for index in range(image_count):
        filename = f"fixture-{index}.jpg"
        evidence = []
        if include_detection:
            evidence.append(visual_evidence(context, filename))
        images.append(
            ImageVisualAssessment(
                damage_assessment=ImageDamageAssessment(
                    filename=filename,
                    width=100,
                    height=100,
                    detections=[context.damage] if include_detection else [],
                ),
                part_assessment=ImagePartAssessment(
                    filename=filename,
                    width=100,
                    height=100,
                    detections=[context.part],
                    model_metadata=context.part_model,
                    confidence_threshold=0.25,
                ),
                evidence=evidence,
            )
        )
    return images


def visual_evidence(context, filename):
    return VisualDamageEvidence(
        filename=filename,
        asset_reference=f"fixture/images/{filename}",
        damage_detection_index=0,
        visual_damage_type=VisualDamageType.SCRATCH,
        damage_confidence=0.87,
        vehicle_part=VehiclePartType.HOOD,
        part_confidence=0.94,
        containment_ratio=1,
        mapped_damage_type="HOOD",
        association_status="assigned",
        damage_detection=context.damage,
        primary_association=context.association,
        associations=[context.association],
        damage_model_metadata=context.damage_model,
        part_model_metadata=context.part_model,
        fusion_policy=DEFAULT_FUSION_POLICY,
    )
