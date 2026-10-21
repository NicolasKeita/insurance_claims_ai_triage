"""Test overlap."""

import pytest

from claims.enums import DamageType
from tests.vision.fusion.association._support import (
    damage,
    fused,
    part,
    polygon_for,
    rectangle,
)
from vision.fusion import calculate_overlap
from vision.models import (
    AssociationMethod,
    AssociationStatus,
)


def test_iou_trap_small_damage_inside_large_part_is_assigned_by_containment():
    result = fused([damage()], [part()])
    association = result.evidence[0].primary_association
    assert association.damage_area == pytest.approx(0.01)
    assert association.part_area == pytest.approx(1.0)
    assert association.intersection_area == pytest.approx(0.01)
    assert association.damage_containment_ratio == pytest.approx(1.0)
    assert association.iou == pytest.approx(0.01)
    assert association.association_method is AssociationMethod.MASK
    assert result.evidence[0].association_status is AssociationStatus.ASSIGNED
    assert result.evidence[0].mapped_damage_type is DamageType.HOOD


@pytest.mark.parametrize(
    "part_bounds,containment",
    [
        (rectangle(0.15, 0.1, 0.25, 0.2), 0.5),
        (rectangle(0.3, 0.3, 0.5, 0.5), 0.0),
    ],
)
def test_half_and_zero_overlap(part_bounds, containment):
    result = calculate_overlap(damage(), part(part_bounds))
    assert result.damage_containment_ratio == pytest.approx(containment)
    assert result.intersection_area == pytest.approx(result.damage_area * containment)


def test_masks_take_precedence_over_overlapping_boxes():
    """
    Both boxes cover the image, but the observed mask regions are disjoint.
    """
    full_box = rectangle(0, 0, 1, 1)
    observation = damage(full_box).model_copy(update={"polygon": polygon_for(rectangle(0, 0, 0.2, 0.2))})
    vehicle_part = part(full_box).model_copy(update={"polygon": polygon_for(rectangle(0.8, 0.8, 1, 1))})
    overlap = calculate_overlap(observation, vehicle_part)
    assert overlap.association_method is AssociationMethod.MASK
    assert overlap.damage_containment_ratio == 0
    assert fused([observation], [vehicle_part]).evidence[0].association_status is AssociationStatus.UNASSIGNED
