"""Test ranking."""

import pytest

from tests.vision.fusion.association._support import damage, fused, part, rectangle
from vision.models import (
    AssociationStatus,
)
from vision.taxonomy import (
    VehiclePartType,
)


def test_highest_containment_selects_primary_preserving_other_candidates():
    result = fused(
        [damage(rectangle(0.1, 0.1, 0.3, 0.3))],
        [
            part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
            part(kind=VehiclePartType.HOOD),
        ],
    )
    evidence = result.evidence[0]
    assert evidence.association_status is AssociationStatus.ASSIGNED
    assert evidence.vehicle_part is VehiclePartType.HOOD
    assert [candidate.damage_containment_ratio for candidate in evidence.associations] == pytest.approx([1, 0.8])


def test_close_eligible_candidates_remain_ambiguous_without_forced_business_mapping():
    result = fused(
        [damage(rectangle(0.1, 0.1, 0.3, 0.3))],
        [
            part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
            part(rectangle(0.1, 0.1, 0.265, 0.3), kind=VehiclePartType.HOOD),
        ],
    )
    evidence = result.evidence[0]
    assert evidence.association_status is AssociationStatus.AMBIGUOUS
    assert len(evidence.associations) == 2
    assert evidence.primary_association is None
    assert evidence.vehicle_part is None and evidence.mapped_damage_type is None


@pytest.mark.parametrize(
    "observation,vehicle_part",
    [
        (damage(confidence=0.49), part()),
        (damage(), part(confidence=0.49)),
        (damage(), part(rectangle(0.15, 0.1, 0.3, 0.3))),
        (damage(), part(rectangle(0.8, 0.8, 1, 1))),
    ],
)
def test_unreliable_or_uncovered_damage_is_unassigned(observation, vehicle_part):
    evidence = fused([observation], [vehicle_part]).evidence[0]
    assert evidence.association_status is AssociationStatus.UNASSIGNED
    assert evidence.mapped_damage_type is None


def test_low_confidence_best_overlap_does_not_hide_eligible_part():
    result = fused(
        [damage(rectangle(0.1, 0.1, 0.3, 0.3))],
        [
            part(kind=VehiclePartType.HOOD, confidence=0.4),
            part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
        ],
    )
    assert result.evidence[0].vehicle_part is VehiclePartType.FRONT_BUMPER
    assert len(result.evidence[0].associations) == 2
    assert result.evidence[0].associations[0].eligible is False
