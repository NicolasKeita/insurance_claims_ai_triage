"""Contours."""

from __future__ import annotations

from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf._constants import Point
from vision.dataset.cardd_hf.masks._geometry import _area, _bridge, _contains


def _trace_rings(edges):
    directions = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    rings: list[list[Point]] = []
    while edges:
        start = min(edges)
        current = start
        direction = None
        ring: list[Point] = []
        while True:
            ring.append(current)
            choices = edges.get(current)
            if not choices:
                raise DatasetValidationError("Broken instance mask boundary")
            if direction is None or len(choices) == 1:
                following = min(choices)
            else:
                preference = {3: 0, 0: 1, 1: 2, 2: 3}
                following = min(
                    choices,
                    key=lambda p: preference[(directions[(p[0] - current[0], p[1] - current[1])] - direction) % 4],
                )
            direction = directions[(following[0] - current[0], following[1] - current[1])]
            choices.remove(following)
            if not choices:
                del edges[current]
            current = following
            if current == start:
                break
        simplified = [
            point
            for i, point in enumerate(ring)
            if (point[0] - ring[i - 1][0]) * (ring[(i + 1) % len(ring)][1] - point[1])
            != (point[1] - ring[i - 1][1]) * (ring[(i + 1) % len(ring)][0] - point[0])
        ]
        rings.append(simplified)
    return rings


def _bridge_holes(rings):
    outers = [ring for ring in rings if _area(ring) > 0]
    holes = [ring for ring in rings if _area(ring) < 0]
    for hole in holes:
        a, b = hole[0], hole[1]
        delta = (b[0] - a[0], b[1] - a[1])
        length = max(abs(delta[0]), abs(delta[1]))
        point = ((a[0] + b[0]) / 2 + delta[1] / length * 0.01, (a[1] + b[1]) / 2 - delta[0] / length * 0.01)
        enclosing = [i for i, outer in enumerate(outers) if _contains(outer, point)]
        if not enclosing:
            raise DatasetValidationError("Instance mask hole has no enclosing component")
        index = min(enclosing, key=lambda i: abs(_area(outers[i])))
        outers[index] = _bridge(outers[index], hole)
    return outers, holes
