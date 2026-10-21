"""Polygons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vision.dataset._models import DatasetValidationError


def _polygon(annotation: dict[str, Any], width: int, height: int) -> tuple[list[float] | None, str | None]:
    """
    YOLO stores one polygon per instance. Preserve every component vertex and retrace each connector so it
    adds zero polygon area. Matching winding keeps component areas additive. This cannot preserve COCO
    disconnected topology exactly after rasterization; the manifest records the representation limit.
    """
    if annotation.get("iscrowd", 0):
        return None, "crowd"
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        return None, "rle"
    if not segmentation:
        return None, "missing_segmentation"
    if not isinstance(segmentation, list) or any(not isinstance(part, list) for part in segmentation):
        return None, "invalid_polygon"
    parts = []
    areas = []
    for values in segmentation:
        points, area, reason = _polygon_component(values, width, height)
        if reason:
            return None, reason
        parts.append(points)
        areas.append(area)
    points = parts[0]
    for component, area in zip(parts[1:], areas[1:]):
        if area * areas[0] < 0:
            component = component[::-1]
        _, left, right = min(
            ((x - a) ** 2 + (y - b) ** 2, i, j) for i, (x, y) in enumerate(points) for j, (a, b) in enumerate(component)
        )
        rotated = component[right:] + component[:right]
        points = points[: left + 1] + rotated + [rotated[0], points[left]] + points[left + 1 :]
    return [value for x, y in points for value in (x / width, y / height)], None


def _polygon_component(values, width, height):
    if len(values) < 6 or len(values) % 2:
        return None, None, "invalid_polygon"
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
    ):
        return None, None, "invalid_coordinates"
    points = [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]
    if any(not 0 <= x <= width or not 0 <= y <= height for x, y in points):
        return None, None, "invalid_coordinates"
    if points[0] == points[-1]:
        points.pop()
    area = (
        sum(
            x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
            for i, (x, y) in enumerate(points)
        )
        if points
        else 0
    )
    if len(set(points)) < 3 or abs(area) < 1e-9:
        return None, None, "degenerate_polygon"
    return points, area, None


def validate_segmentation_label(
    path: Path, *, class_count: int, expected_count: int | None = None
) -> list[tuple[int, list[float]]]:
    """Validate YOLO polygon labels; a five-field bounding box is never accepted."""
    if not path.is_file():
        raise DatasetValidationError(f"Missing prepared label: {path}")
    rows = []
    seen_rows = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise DatasetValidationError(f"Blank segmentation row at {path}:{line_number}")
        fields = line.split()
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except (ValueError, IndexError) as exc:
            raise DatasetValidationError(f"Invalid segmentation row at {path}:{line_number}") from exc
        if not 0 <= class_id < class_count or len(coordinates) < 6 or len(coordinates) % 2:
            raise DatasetValidationError(f"Invalid class ID or polygon vertex count at {path}:{line_number}")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise DatasetValidationError(
                f"Polygon coordinates must be finite and normalized to [0,1] at {path}:{line_number}"
            )
        polygon, reason = _polygon({"segmentation": [coordinates]}, 1, 1)
        if reason or polygon is None:
            raise DatasetValidationError(f"Invalid polygon ({reason}) at {path}:{line_number}")
        key = (class_id, *coordinates)
        if key in seen_rows:
            raise DatasetValidationError(f"Duplicate segmentation row would be removed by YOLO at {path}:{line_number}")
        seen_rows.add(key)
        rows.append((class_id, coordinates))
    if expected_count is not None and len(rows) != expected_count:
        raise DatasetValidationError(f"Unexpected label count for {path}: expected {expected_count}, found {len(rows)}")
    return rows
