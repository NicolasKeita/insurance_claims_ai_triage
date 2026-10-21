"""Test geometry."""

import pytest
from pydantic import ValidationError

from tests.intelligence.vision.fusion.association._support import damage, fused, part
from vision.fusion import FusionGeometryError, calculate_overlap
from vision.models import (
    NormalizedPoint,
    NormalizedPolygon,
)


@pytest.mark.parametrize(
    "coordinates",
    [
        [(0, 0), (1, 1), (0, 1), (1, 0)],
        [(0.1, 0.1), (0.9, 0.8), (0.1, 0.9), (0.8, 0.1)],
    ],
)
def test_invalid_self_crossing_polygon_rejected(coordinates):
    """
    Step29 already rejects signed-zero area geometry.
    """
    points = [NormalizedPoint(x=x, y=y) for x, y in coordinates]
    try:
        polygon = NormalizedPolygon(points=points)
    except ValidationError:
        return
    with pytest.raises(FusionGeometryError, match="fallback is forbidden"):
        calculate_overlap(damage().model_copy(update={"polygon": polygon}), part())


def test_step29_retraced_nonzero_ring_can_load_but_is_rejected_for_fusion():
    """
    Ultralytics xyn can bridge contours using a retraced edge. Preserve the step29 artifact loading
    contract; fusion must reject its invalid topology.
    """
    polygon = NormalizedPolygon(
        points=[
            NormalizedPoint(x=x, y=y)
            for x, y in [
                (0.1, 0.1),
                (0.3, 0.1),
                (0.3, 0.3),
                (0.1, 0.3),
                (0.1, 0.1),
                (0.5, 0.1),
                (0.7, 0.1),
                (0.7, 0.3),
                (0.5, 0.3),
                (0.5, 0.1),
                (0.1, 0.1),
            ]
        ]
    )
    with pytest.raises(FusionGeometryError, match="fallback is forbidden"):
        calculate_overlap(damage().model_copy(update={"polygon": polygon}), part())


@pytest.mark.parametrize("part_mask", [True, False])
def test_invalid_geometry_never_silently_becomes_bbox_even_if_other_mask_missing(part_mask):
    invalid = NormalizedPolygon.model_construct(
        points=[NormalizedPoint(x=x, y=y) for x, y in [(0, 0), (1, 1), (0, 1), (1, 0)]]
    )
    observation = damage().model_copy(update={"polygon": invalid})
    with pytest.raises(FusionGeometryError, match="fallback is forbidden"):
        calculate_overlap(observation, part(mask=part_mask))
    with pytest.raises(FusionGeometryError):
        fused([observation], [])
