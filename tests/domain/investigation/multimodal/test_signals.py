"""Test signals."""

from claims.enums import DamageType
from claims.investigation import (
    DEFAULT_INVESTIGATION_POLICY,
    INVESTIGATION_POLICY_V1,
    INVESTIGATION_POLICY_V2,
    INVESTIGATION_POLICY_V3,
    SignalCode,
)
from tests.domain.investigation.multimodal._assessment import (
    investigate,
    make_visual_assessment,
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
