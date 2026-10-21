"""Test thresholds."""

from tests.intelligence.vision.fusion.association._support import damage, fused, part, rectangle
from vision.fusion_config import VisionFusionPolicy
from vision.models import (
    AssociationStatus,
)
from vision.taxonomy import (
    VehiclePartType,
)


def test_thresholds_and_ambiguity_are_configurable():
    observation = damage(rectangle(0.1, 0.1, 0.3, 0.3), confidence=0.4)
    vehicle_part = part(rectangle(0.2, 0.1, 0.3, 0.3), confidence=0.4)
    assert fused([observation], [vehicle_part]).evidence[0].association_status is AssociationStatus.UNASSIGNED
    policy = VisionFusionPolicy(min_damage_confidence=0.3, min_part_confidence=0.3, min_damage_containment=0.5)
    assert (
        fused([observation], [vehicle_part], policy=policy).evidence[0].association_status is AssociationStatus.ASSIGNED
    )
    assert VisionFusionPolicy.from_toml("configs/vision/vision_fusion_v1.toml") == VisionFusionPolicy()


def test_narrower_ambiguity_margin_can_resolve_close_candidates():
    observation = damage(rectangle(0.1, 0.1, 0.3, 0.3))
    parts = [
        part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
        part(rectangle(0.1, 0.1, 0.265, 0.3), kind=VehiclePartType.HOOD),
    ]
    assert fused([observation], parts).evidence[0].association_status is AssociationStatus.AMBIGUOUS
    evidence = fused([observation], parts, policy=VisionFusionPolicy(ambiguity_margin=0.01)).evidence[0]
    assert evidence.association_status is AssociationStatus.ASSIGNED
    assert evidence.vehicle_part is VehiclePartType.HOOD
