"""Lossless binary pixel-cell geometry in the existing normalized polygon model.

Ultralytics' native-resolution masks already undo letterboxing. Shapely unions
foreground pixel runs; unlike an external contour this preserves thin regions,
holes and disconnected components without repairing or guessing any geometry.
"""

from vision.masks._polygon import BinaryMaskGeometryError, binary_mask_to_polygon, native_mask_polygons

__all__ = ["BinaryMaskGeometryError", "binary_mask_to_polygon", "native_mask_polygons"]
