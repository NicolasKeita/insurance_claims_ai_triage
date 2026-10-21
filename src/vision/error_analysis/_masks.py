"""Masks."""

from __future__ import annotations


def polygon_mask_iou(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
) -> float:
    return polygon_mask_overlap(first, second, image_size)["iou"]


def polygon_mask_overlap(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
) -> dict[str, float]:
    """Diagnostic overlap of normalized polygons, rasterized with Pillow.

    This does not recreate Ultralytics' resized prototype masks or AP matcher.
    Missing polygons have zero overlap. Multi-component bridge edges and hole
    boundaries follow the prepared YOLO polygon representation.
    """
    from PIL import Image, ImageChops, ImageDraw

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Diagnostic mask size must be positive")
    masks = [Image.new("1", image_size) for _ in range(2)]
    try:
        for polygon, mask in zip((first, second), masks):
            if len(polygon) >= 3:
                ImageDraw.Draw(mask).polygon([(x * width, y * height) for x, y in polygon], fill=1)
        intersection = ImageChops.logical_and(*masks)
        union = ImageChops.logical_or(*masks)
        try:
            intersection_count = sum(intersection.histogram()[1:])
            union_count = sum(union.histogram()[1:])
            truth_count = sum(masks[1].histogram()[1:])
            return {
                "iou": intersection_count / union_count if union_count else 0.0,
                "truth_coverage": intersection_count / truth_count if truth_count else 0.0,
            }
        finally:
            intersection.close()
            union.close()
    finally:
        for mask in masks:
            mask.close()
