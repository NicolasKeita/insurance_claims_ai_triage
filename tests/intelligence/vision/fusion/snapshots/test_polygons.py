"""Test polygons."""

import pytest

from tests.intelligence.vision.fusion.snapshots._support import points
from vision.geometry import PolygonGeometryError, validated_polygon_geometry
from vision.models import NormalizedPolygon


def test_polygon_representation_preserves_holes_and_disconnected_components():
    polygon = NormalizedPolygon(
        points=points(0, 0, 0.5, 0.5),
        holes=[points(0.1, 0.1, 0.2, 0.2)],
        components=[NormalizedPolygon(points=points(0.8, 0.8, 1, 1))],
    )
    geometry = validated_polygon_geometry(polygon)
    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == pytest.approx(0.25 - 0.01 + 0.04)
    assert len(geometry.geoms[0].interiors) == 1
    assert NormalizedPolygon.model_validate(polygon.model_dump(round_trip=True)) == polygon


def test_overlapping_components_and_hole_outside_exterior_are_rejected():
    for polygon in [
        NormalizedPolygon(
            points=points(0, 0, 0.5, 0.5), components=[NormalizedPolygon(points=points(0.2, 0.2, 0.8, 0.8))]
        ),
        NormalizedPolygon(points=points(0, 0, 0.5, 0.5), holes=[points(0.6, 0.6, 0.7, 0.7)]),
    ]:
        with pytest.raises(PolygonGeometryError, match="Invalid polygon topology"):
            validated_polygon_geometry(polygon)
