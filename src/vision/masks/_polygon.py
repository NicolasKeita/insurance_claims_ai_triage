"""Pixel-run union of a binary mask into the normalized polygon model."""

from typing import Any

from vision.models import NormalizedPoint, NormalizedPolygon


class BinaryMaskGeometryError(ValueError):
    pass


def _validated_mask(mask: Any) -> Any:
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or not all(array.shape):
        raise BinaryMaskGeometryError("A binary mask must have positive height and width")
    if array.dtype.kind not in "buif" or not np.isfinite(array).all() or not ((array == 0) | (array == 1)).all():
        raise BinaryMaskGeometryError("Mask pixels must be binary, finite zero/one values")
    return array


def _run_rectangles(array: Any, *, width: int, height: int) -> list[Any]:
    """One rectangle per maximal run of equal pixels in each row."""
    import numpy as np
    from shapely.geometry import box

    rectangles: list[Any] = []
    active: dict[tuple[int, int], int] = {}
    for row_index, row in enumerate(array.astype(bool, copy=False)):
        changes = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        runs = {tuple(map(int, pair)) for pair in changes.reshape(-1, 2)}
        for start, end in active.keys() - runs:
            rectangles.append(box(start / width, active[(start, end)] / height, end / width, row_index / height))
        active = {run: active.get(run, row_index) for run in runs}
    for (start, end), top in active.items():
        rectangles.append(box(start / width, top / height, end / width, 1.0))
    return rectangles


def _ring_points(ring: Any) -> list[NormalizedPoint]:
    return [NormalizedPoint(x=x, y=y) for x, y in list(ring.coords)[:-1]]


def _components(geometry: Any) -> list[NormalizedPolygon]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    polygons.sort(key=lambda polygon: (-polygon.area, polygon.bounds))
    return [
        NormalizedPolygon(
            points=_ring_points(polygon.exterior),
            holes=[_ring_points(hole) for hole in polygon.interiors],
        )
        for polygon in polygons
    ]


def binary_mask_to_polygon(mask: Any) -> NormalizedPolygon | None:
    """Coalesce equal runs in adjacent rows.

    GEOS, rather than hand-written ring tracing, performs the complete polygon
    union and topology construction.
    """
    from shapely import union_all

    array = _validated_mask(mask)
    height, width = array.shape
    rectangles = _run_rectangles(array, width=width, height=height)
    if not rectangles:
        return None
    geometry = union_all(rectangles).simplify(0, preserve_topology=True)
    if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise BinaryMaskGeometryError("Binary pixel union did not produce valid polygon geometry")
    components = _components(geometry)
    return components[0].model_copy(update={"components": components[1:]})


def native_mask_polygons(data: Any, *, count: int, width: int, height: int) -> list[NormalizedPolygon | None]:
    """Accept detached tensors or arrays, never import a model/framework.

    A mismatched shape is an explicit error: adapters request retina_masks=True
    so Ultralytics performs its official letterbox-aware native mask scaling.
    """
    import numpy as np

    for method in ("detach", "cpu", "numpy"):
        if hasattr(data, method):
            data = getattr(data, method)()
    masks = np.asarray(data)
    if masks.shape != (count, height, width):
        raise BinaryMaskGeometryError(
            f"Binary mask shape {masks.shape} must match ({count}, {height}, {width}); "
            "request native image masks with retina_masks=True, never mix padded mask and image coordinates"
        )
    return [binary_mask_to_polygon(mask) for mask in masks]
