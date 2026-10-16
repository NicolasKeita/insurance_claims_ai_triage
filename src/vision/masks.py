"""Lossless binary pixel-cell geometry in the existing normalized polygon model.

Ultralytics' native-resolution masks already undo letterboxing. Shapely unions
foreground pixel runs; unlike an external contour this preserves thin regions,
holes and disconnected components without repairing or guessing any geometry.
"""

from typing import Any

from vision.models import NormalizedPoint, NormalizedPolygon


class BinaryMaskGeometryError(ValueError):
    pass


def binary_mask_to_polygon(mask: Any) -> NormalizedPolygon | None:
    import numpy as np
    from shapely import union_all
    from shapely.geometry import box

    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape):
        raise BinaryMaskGeometryError("A binary mask must have positive height and width")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise BinaryMaskGeometryError("Mask pixels must be binary, finite zero/one values")
    height, width = mask.shape
    rectangles, active = [], {}

    def rectangle(start, end, top, bottom):
        return box(start / width, top / height, end / width, bottom / height)

    # Coalesce equal runs in adjacent rows. GEOS, rather than hand-written ring
    # tracing, performs the complete polygon union and topology construction.
    for row_index, row in enumerate(mask.astype(bool, copy=False)):
        changes = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        runs = {tuple(map(int, pair)) for pair in changes.reshape(-1, 2)}
        for run in active.keys() - runs:
            rectangles.append(rectangle(*run, active[run], row_index))
        active = {run: active.get(run, row_index) for run in runs}
    rectangles.extend(rectangle(*run, top, height) for run, top in active.items())
    if not rectangles:
        return None
    geometry = union_all(rectangles).simplify(0, preserve_topology=True)
    if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise BinaryMaskGeometryError("Binary pixel union did not produce valid polygon geometry")
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    polygons.sort(key=lambda polygon: (-polygon.area, polygon.bounds))

    def points(ring):
        return [NormalizedPoint(x=x, y=y) for x, y in list(ring.coords)[:-1]]

    components = [
        NormalizedPolygon(points=points(polygon.exterior), holes=[points(hole) for hole in polygon.interiors])
        for polygon in polygons
    ]
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
