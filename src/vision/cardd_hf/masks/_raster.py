"""Mask."""

from __future__ import annotations

import math
from typing import Any

from vision.cardd_hf._constants import Polygon
from vision.cardd_hf.masks._contours import _bridge_holes, _trace_rings
from vision.cardd_hf.masks._geometry import _polygon
from vision.dataset import DatasetValidationError


def _mask_polygons(mask: Any, bbox: Any) -> tuple[tuple[Polygon, ...], int]:
    """Trace binary pixel-cell edges, preserving components and hole vertices.

    Holes become retraced bridges in their outer component, matching YOLO's
    single-contour contract. A rasterizer may add bridge pixels; this limitation
    is explicit in inspection metadata. No mask is replaced by its rectangle.

    Left turn first joins corner-touching foreground (8-connectivity, matching common instance-mask contour
    implementations).

    A point just inside the hole avoids exact boundary ambiguities.
    """
    binary, x, y, width, height = _binary_mask(mask, bbox)
    edges = _boundary_edges(binary)
    rings = _trace_rings(edges)
    outers, holes = _bridge_holes(rings)
    mask_height, mask_width = binary.shape
    polygons = tuple(
        _polygon([(min(1.0, x + px / mask_width * width), min(1.0, y + py / mask_height * height)) for px, py in ring])
        for ring in outers
    )
    return polygons, len(holes)


def _binary_mask(mask, bbox):
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or not array.size:
        raise DatasetValidationError("Instance mask must be a nonempty 2D binary array")
    if (
        not (np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.number))
        or not np.isfinite(array).all()
    ):
        raise DatasetValidationError("Instance mask must contain finite binary pixels")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1, 255}) or (1 in unique and 255 in unique):
        raise DatasetValidationError("Per-instance masks require binary pixels, not semantic class IDs")
    binary = array != 0
    if not binary.any():
        raise DatasetValidationError("Instance mask contains no foreground pixels")
    if (
        not isinstance(bbox, (tuple, list))
        or len(bbox) != 4
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
    ):
        raise DatasetValidationError("Instance mask requires a finite normalized x/y/width/height bounding box")
    x, y, width, height = map(float, bbox)
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise DatasetValidationError("Instance mask bounding box lies outside normalized image coordinates")
    return binary, x, y, width, height


def _boundary_edges(binary):
    import numpy as np

    padded = np.pad(binary, 1)
    boundaries = (
        binary & ~padded[:-2, 1:-1],
        binary & ~padded[1:-1, 2:],
        binary & ~padded[2:, 1:-1],
        binary & ~padded[1:-1, :-2],
    )
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for side, boundary in enumerate(boundaries):
        for row, col in zip(*np.nonzero(boundary)):
            a, b = (
                ((col, row), (col + 1, row))
                if side == 0
                else ((col + 1, row), (col + 1, row + 1))
                if side == 1
                else ((col + 1, row + 1), (col, row + 1))
                if side == 2
                else ((col, row + 1), (col, row))
            )
            edges.setdefault(a, []).append(b)
    return edges
