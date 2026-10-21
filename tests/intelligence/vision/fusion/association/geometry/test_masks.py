"""Test masks."""

import pytest

from tests.intelligence.vision.fusion.association._support import (
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
    NormalizedPolygon,
)


def test_damage_in_a_part_mask_hole_has_zero_containment_despite_the_outer_box():
    vehicle_part = part().model_copy(
        update={
            "polygon": NormalizedPolygon(
                points=polygon_for(rectangle(0, 0, 1, 1)).points,
                holes=[polygon_for(rectangle(0.05, 0.05, 0.25, 0.25)).points],
            )
        }
    )
    overlap = calculate_overlap(damage(), vehicle_part)
    assert overlap.association_method is AssociationMethod.MASK
    assert overlap.damage_containment_ratio == 0
    assert overlap.part_area == pytest.approx(0.96)
    assert fused([damage()], [vehicle_part]).evidence[0].association_status is AssociationStatus.UNASSIGNED


def test_disconnected_part_component_can_support_a_contained_damage():
    vehicle_part = part().model_copy(
        update={
            "polygon": NormalizedPolygon(
                points=polygon_for(rectangle(0.7, 0.7, 1, 1)).points,
                components=[polygon_for(rectangle(0, 0, 0.3, 0.3))],
            )
        }
    )
    evidence = fused([damage()], [vehicle_part]).evidence[0]
    assert evidence.association_status is AssociationStatus.ASSIGNED
    assert evidence.containment_ratio == pytest.approx(1)
    assert evidence.primary_association.part_area == pytest.approx(0.18)


@pytest.mark.parametrize("damage_mask,part_mask", [(False, True), (True, False), (False, False)])
def test_only_missing_masks_use_explicit_bbox_fallback(damage_mask, part_mask):
    overlap = calculate_overlap(damage(mask=damage_mask), part(mask=part_mask))
    assert overlap.damage_containment_ratio == pytest.approx(1)
    assert overlap.association_method is AssociationMethod.BBOX_FALLBACK
