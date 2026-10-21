"""Test models."""

import pytest
from pydantic import ValidationError

from claims.enums import DamageType
from tests.intelligence.vision.perception.domain._support import detection
from vision.models import (
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
)
from vision.taxonomy import VisualDamageType, map_category_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("dent", VisualDamageType.DENT),
        ("Scratch", VisualDamageType.SCRATCH),
        ("crack", VisualDamageType.CRACK),
        ("glass shatter", VisualDamageType.GLASS_SHATTER),
        ("lamp_broken", VisualDamageType.LAMP_BROKEN),
        ("tire-flat", VisualDamageType.TIRE_FLAT),
    ],
)
def test_damage_mapping_uses_names_not_framework_ids(name, expected):
    assert map_category_name(name) == expected


@pytest.mark.parametrize("name", ["car", "bumper", "LEFT_HEADLIGHT", "shattered windshield", "", None])
def test_unknown_names_and_vehicle_parts_are_rejected(name):
    with pytest.raises(ValueError):
        map_category_name(name)
    assert VisualDamageType.LAMP_BROKEN.value != DamageType.LEFT_HEADLIGHT.value


@pytest.mark.parametrize(
    "changes",
    [
        {"x_min": -0.1},
        {"x_max": 1.1},
        {"y_min": float("nan")},
        {"x_min": 0.9, "x_max": 0.2},
        {"y_min": 0.5, "y_max": 0.5},
    ],
)
def test_invalid_boxes_fail(changes):
    fields = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}
    with pytest.raises(ValidationError):
        NormalizedBoundingBox(**(fields | changes))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_confidence_is_bounded(confidence):
    with pytest.raises(ValidationError):
        detection(confidence=confidence)


def test_polygon_rejects_degenerate_or_unbounded_points():
    with pytest.raises(ValidationError):
        NormalizedPolygon(points=[NormalizedPoint(x=0.1, y=0.1)] * 3)
    with pytest.raises(ValidationError):
        NormalizedPoint(x=1.1, y=0.2)
