"""Test taxonomy."""

import pytest
from pydantic import ValidationError

from claims.enums import DamageType
from tests.vision.fusion.association._support import damage, fused, part
from vision.models import (
    AssociationStatus,
)
from vision.taxonomy import (
    VehiclePartType,
    map_vehicle_part_category_name,
    map_vehicle_part_to_damage,
)


@pytest.mark.parametrize(
    "kind,expected",
    [
        (VehiclePartType.FRONT_BUMPER, DamageType.FRONT_BUMPER),
        (VehiclePartType.HOOD, DamageType.HOOD),
        (VehiclePartType.FRONT_LEFT_LIGHT, DamageType.LEFT_HEADLIGHT),
        (VehiclePartType.FRONT_LIGHT, None),
        (VehiclePartType.FRONT_RIGHT_LIGHT, None),
        (VehiclePartType.FRONT_DOOR, None),
        (VehiclePartType.OTHER, None),
    ],
)
def test_business_mapping_is_intentionally_limited(kind, expected):
    assert map_vehicle_part_to_damage(kind) == expected


def test_dataset_object_remains_other_and_cannot_become_a_business_part():
    assert map_vehicle_part_category_name("object") is VehiclePartType.OTHER
    evidence = fused([damage()], [part(kind=VehiclePartType.OTHER)]).evidence[0]
    assert evidence.association_status is AssociationStatus.UNASSIGNED
    assert evidence.associations == []
    assert not hasattr(DamageType, "OBJECT")


def test_assigned_snapshot_cannot_be_tampered_into_a_different_business_part():
    evidence = fused([damage()], [part()]).evidence[0]
    data = evidence.model_dump(round_trip=True)
    data["mapped_damage_type"] = DamageType.LEFT_HEADLIGHT
    with pytest.raises(ValidationError, match="business mapping"):
        type(evidence).model_validate(data)
