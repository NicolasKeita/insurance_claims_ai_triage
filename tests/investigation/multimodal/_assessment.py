"""Assessment."""

from claims.enums import DamageType
from investigation import (
    INVESTIGATION_POLICY_V3,
    InvestigationContext,
    assess_investigation,
)
from tests.investigation.multimodal._images import visual_images
from tests.investigation.policy._support import (
    make_anomaly,
    make_consistency,
    make_features,
)
from vision.fusion_config import DEFAULT_FUSION_POLICY
from vision.models import (
    ClaimMultimodalVisionAssessment,
    MultimodalConsistencyAssessment,
    VisionModelMetadata,
)


def make_visual_assessment(
    *,
    claim_id="CLAIM-2026-00001",
    declared=(DamageType.FRONT_BUMPER,),
    include_detection=True,
    image_count=1,
):
    damage_model = VisionModelMetadata(identifier="fixture-damage", framework="fake", version="damage-v1")
    part_model = VisionModelMetadata(identifier="fixture-parts", framework="fake", version="parts-v1")
    images = visual_images(image_count, include_detection, damage_model, part_model)
    mapped = {DamageType.HOOD} if include_detection and image_count else set()
    return ClaimMultimodalVisionAssessment(
        claim_id=claim_id,
        images=images,
        damage_model_metadata=damage_model,
        part_model_metadata=part_model,
        fusion_policy=DEFAULT_FUSION_POLICY,
        consistency=MultimodalConsistencyAssessment(
            declared_damage_types=list(declared),
            quoted_damage_types=[DamageType.HOOD],
            visually_supported_declared_damage=sorted(set(declared) & mapped, key=lambda v: v.value),
            additional_visual_damage=sorted(mapped - set(declared), key=lambda v: v.value),
            visually_supported_quote_damage=list(mapped),
            declared_damage_not_visually_confirmed=sorted(set(declared) - mapped, key=lambda v: v.value),
        ),
    )


def investigate(visual, policy=INVESTIGATION_POLICY_V3):
    return assess_investigation(
        consistency=make_consistency(),
        anomaly=make_anomaly(),
        features=make_features(),
        policy=policy,
        context=InvestigationContext(visual_assessment=visual),
    )
