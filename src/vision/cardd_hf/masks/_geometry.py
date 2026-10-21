"""Geometry."""

from __future__ import annotations

import math
from typing import Any

from vision.cardd_hf._constants import Point, Polygon
from vision.dataset import DatasetValidationError


def _area(points: Polygon | list[Point]) -> float:
    return (
        sum(
            x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
            for i, (x, y) in enumerate(points)
        )
        / 2
    )


def _polygon(points: Any) -> Polygon:
    if not isinstance(points, (tuple, list)):
        raise DatasetValidationError("Instance polygon points must be a sequence")
    result: list[Point] = []
    for point in points:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise DatasetValidationError("Instance polygon vertices must be x/y pairs")
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in point
        ):
            raise DatasetValidationError("Instance polygons require finite normalized coordinates in [0, 1]")
        result.append((float(point[0]), float(point[1])))
    if result and result[-1] == result[0]:
        result.pop()
    if len(set(result)) < 3 or abs(_area(result)) < 1e-12:
        raise DatasetValidationError("Degenerate instance polygon")
    return tuple(result)


def _contains(polygon: Polygon | list[Point], point: Point) -> bool:
    x, y = point
    inside = False
    for i, (a, b) in enumerate(polygon):
        c, d = polygon[(i + 1) % len(polygon)]
        if (b > y) != (d > y) and x < (c - a) * (y - b) / (d - b) + a:
            inside = not inside
    return inside


def _bridge(outer: list[Point], ring: list[Point]) -> list[Point]:
    """Retrace a zero-area connector; retain the hole ring's opposite winding."""
    _, left, right = min(
        ((x - a) ** 2 + (y - b) ** 2, i, j) for i, (x, y) in enumerate(outer) for j, (a, b) in enumerate(ring)
    )
    rotated = ring[right:] + ring[:right]
    return outer[: left + 1] + rotated + [rotated[0], outer[left]] + outer[left + 1 :]
