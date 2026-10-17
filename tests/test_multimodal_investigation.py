"""Checkpoint-free policy checks using reproducible per-photo evidence fixtures."""

import pytest
from pydantic import ValidationError
from test_investigation import make_anomaly, make_consistency, make_features

from claims.enums import DamageType
from investigation import (
    DEFAULT_INVESTIGATION_POLICY,
    INVESTIGATION_POLICY_V1,
    INVESTIGATION_POLICY_V2,
    INVESTIGATION_POLICY_V3,
    InvestigationContext,
    SignalCode,
    assess_investigation,
)
from investigation.signals import signals_from_multimodal
from vision.fusion_config import DEFAULT_FUSION_POLICY
from vision.models import (
    ClaimMultimodalVisionAssessment,
    DamagePartAssociation,
    ImageDamageAssessment,
    ImagePartAssessment,
    ImageVisualAssessment,
    MultimodalConsistencyAssessment,
    NormalizedBoundingBox,
    NormalizedPolygon,
    VehiclePartDetection,
    VisionModelMetadata,
    VisualDamageDetection,
    VisualDamageEvidence,
)
from vision.taxonomy import VehiclePartType, VisualDamageType


def make_visual_assessment(
    *,
    claim_id="CLAIM-2026-00001",
    declared=(DamageType.FRONT_BUMPER,),
    include_detection=True,
    image_count=1,
):
    damage_model = VisionModelMetadata(identifier="fixture-damage", framework="fake", version="damage-v1")
    part_model = VisionModelMetadata(identifier="fixture-parts", framework="fake", version="parts-v1")
    damage_box = NormalizedBoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    part_box = NormalizedBoundingBox(x_min=0, y_min=0, x_max=0.5, y_max=0.5)

    def rectangle(box):
        return NormalizedPolygon(
            points=[
                {"x": box.x_min, "y": box.y_min},
                {"x": box.x_max, "y": box.y_min},
                {"x": box.x_max, "y": box.y_max},
                {"x": box.x_min, "y": box.y_max},
            ]
        )

    damage = VisualDamageDetection(
        damage_type=VisualDamageType.SCRATCH,
        confidence=0.87,
        bounding_box=damage_box,
        polygon=rectangle(damage_box),
        model_identifier="fixture-damage",
        model_version="damage-v1",
    )
    part = VehiclePartDetection(
        part_type=VehiclePartType.HOOD,
        confidence=0.94,
        bounding_box=part_box,
        polygon=rectangle(part_box),
        model_identifier="fixture-parts",
        model_version="parts-v1",
    )
    association = DamagePartAssociation(
        damage_detection_index=0,
        part_detection_index=0,
        candidate_part=part,
        damage_confidence=0.87,
        part_confidence=0.94,
        intersection_area=0.01,
        damage_area=0.01,
        part_area=0.25,
        damage_containment_ratio=1,
        iou=0.04,
        association_method="MASK",
        eligible=True,
    )
    images = []
    for index in range(image_count):
        filename = f"fixture-{index}.jpg"
        evidence = []
        if include_detection:
            evidence.append(
                VisualDamageEvidence(
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
                    damage_detection=damage,
                    primary_association=association,
                    associations=[association],
                    damage_model_metadata=damage_model,
                    part_model_metadata=part_model,
                    fusion_policy=DEFAULT_FUSION_POLICY,
                )
            )
        images.append(
            ImageVisualAssessment(
                damage_assessment=ImageDamageAssessment(
                    filename=filename,
                    width=100,
                    height=100,
                    detections=[damage] if include_detection else [],
                ),
                part_assessment=ImagePartAssessment(
                    filename=filename,
                    width=100,
                    height=100,
                    detections=[part],
                    model_metadata=part_model,
                    confidence_threshold=0.25,
                ),
                evidence=evidence,
            )
        )
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


def test_v1_v2_weights_defaults_and_behavior_remain_unchanged():
    expected = {
        SignalCode.DOCUMENT_FIELD_MISMATCH: 10,
        SignalCode.ADDITIONAL_QUOTE_DAMAGE: 18,
        SignalCode.UNMAPPED_QUOTE_ITEM: 8,
        SignalCode.ANOMALOUS_PATTERN: 18,
        SignalCode.EXTREME_ANOMALY_PERCENTILE: 20,
        SignalCode.MISSING_DOCUMENTS: 12,
    }
    assert dict(INVESTIGATION_POLICY_V1.signal_weights) == expected
    assert dict(INVESTIGATION_POLICY_V2.signal_weights) == {
        **expected,
        SignalCode.RECENT_CLAIM_FREQUENCY: 15,
        SignalCode.PREVIOUS_ANOMALOUS_CLAIMS: 18,
    }
    assert DEFAULT_INVESTIGATION_POLICY is INVESTIGATION_POLICY_V2
    for policy in (INVESTIGATION_POLICY_V1, INVESTIGATION_POLICY_V2):
        assert not policy.multimodal_enabled
        assert investigate(make_visual_assessment(), policy).signals == ()
    assert INVESTIGATION_POLICY_V3.version == "investigation_policy_v3"
    assert INVESTIGATION_POLICY_V3.multimodal_enabled
    assert INVESTIGATION_POLICY_V3.signal_weights[SignalCode.VISUAL_ADDITIONAL_DAMAGE] == 12


def test_additional_visual_damage_has_complete_audit_snapshot_and_moderate_score():
    result = investigate(make_visual_assessment())
    assert result.review_score == 12
    assert not result.review_recommended
    assert len(result.signals) == 1
    signal = result.signals[0]
    assert signal.code == SignalCode.VISUAL_ADDITIONAL_DAMAGE
    assert signal.category == "MULTIMODAL_CONSISTENCY"
    assert "not listed among the declared" in signal.explanation
    assert "concealed" not in signal.explanation
    assert signal.evidence["damage_model_metadata"]["version"] == "damage-v1"
    assert signal.evidence["part_model_metadata"]["version"] == "parts-v1"
    assert signal.evidence["fusion_policy_version"] == "vision_fusion_v1"
    snapshot = signal.evidence["observations"][0]
    assert snapshot["filename"] == "fixture-0.jpg"
    assert snapshot["damage_detection"]["polygon"]["points"]
    assert snapshot["primary_association"]["damage_containment_ratio"] == 1
    assert snapshot["primary_association"]["iou"] == 0.04
    assert snapshot["mapped_damage_type"] == "HOOD"


def test_positive_support_and_non_detection_never_create_risk_signals():
    supported = make_visual_assessment(declared=(DamageType.HOOD,))
    assert supported.consistency.visually_supported_declared_damage == [DamageType.HOOD]
    assert supported.consistency.visually_supported_quote_damage == [DamageType.HOOD]
    assert investigate(supported).review_score == 0
    unseen = make_visual_assessment(include_detection=False)
    assert unseen.consistency.declared_damage_not_visually_confirmed == [DamageType.FRONT_BUMPER]
    assert investigate(unseen).signals == ()
    assert "DECLARED_DAMAGE_MISSING_FROM_IMAGE" not in SignalCode.__members__


def test_each_photo_is_preserved_but_does_not_multiply_same_category_weight():
    result = investigate(make_visual_assessment(image_count=2))
    assert result.review_score == 12
    assert len(result.signals) == 1
    assert result.signals[0].evidence["observation_count"] == 2
    assert {v["filename"] for v in result.signals[0].evidence["observations"]} == {
        "fixture-0.jpg",
        "fixture-1.jpg",
    }


def test_mutated_confidence_cannot_create_an_inconsistent_signal_snapshot():
    visual = make_visual_assessment()
    weak = visual.images[0].evidence[0].model_copy(update={"damage_confidence": 0.1})
    visual.images[0].evidence[0] = weak
    with pytest.raises(ValidationError):
        signals_from_multimodal(visual, INVESTIGATION_POLICY_V3)


def test_mutated_additional_summary_cannot_create_an_inconsistent_signal_snapshot():
    visual = make_visual_assessment(declared=(DamageType.HOOD,))
    visual.consistency.additional_visual_damage.append(DamageType.HOOD)
    with pytest.raises(ValidationError):
        signals_from_multimodal(visual, INVESTIGATION_POLICY_V3)
