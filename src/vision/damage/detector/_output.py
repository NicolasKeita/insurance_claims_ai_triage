"""Framework-agnostic extraction and validation of detector output rows."""

from typing import Any

from vision.damage.detector._errors import DetectorOutputError
from vision.models import NormalizedBoundingBox, NormalizedPoint, NormalizedPolygon


def _list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise DetectorOutputError("Detector output must contain sequence values")
    return list(value)


def extract_ultralytics_rows(result: Any, *, width: int, height: int) -> tuple[list, list, list, list | None]:
    """Small shared extraction layer for the two segmentation adapters."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], [], [], None
    coordinates, classes, confidences = _list(boxes.xyxyn), _list(boxes.cls), _list(boxes.conf)
    if not len(coordinates) == len(classes) == len(confidences):
        raise DetectorOutputError("Detector box/class/confidence counts differ")
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is not None:
        from vision.masks import BinaryMaskGeometryError, native_mask_polygons

        try:
            polygons = native_mask_polygons(data, count=len(coordinates), width=width, height=height)
        except BinaryMaskGeometryError as error:
            raise DetectorOutputError(str(error)) from error
    else:
        polygons = list(masks.xyn) if masks is not None else None
    if polygons is not None and len(polygons) != len(coordinates):
        raise DetectorOutputError("Detector mask and box counts differ")
    return coordinates, classes, confidences, polygons


def _ring_area_twice(points: list[NormalizedPoint]) -> float:
    return sum(
        point.x * points[(position + 1) % len(points)].y - points[(position + 1) % len(points)].x * point.y
        for position, point in enumerate(points)
    )


def normalized_detection_geometry(box: Any, contour: Any | None, index: int):
    """Reuse the step29 geometry, retaining explicit missing-mask warnings.

    A retraced thin contour has no area. A self-crossing bowtie encloses real regions after topology repair and
    must be rejected, not boxed.
    """
    if len(box) != 4:
        raise DetectorOutputError("Detector box must contain four normalized coordinates")
    bounding_box = NormalizedBoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3])
    if contour is None:
        return bounding_box, None, []
    if isinstance(contour, NormalizedPolygon):
        return bounding_box, contour, []
    points = _list(contour)
    if any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in points):
        raise DetectorOutputError("Each polygon point must contain x and y")
    normalized_points = [NormalizedPoint(x=point[0], y=point[1]) for point in points]
    if len(points) < 3:
        return bounding_box, None, [f"Detection {index} has no usable segmentation polygon; bounding box retained."]
    if abs(_ring_area_twice(normalized_points)) > 1e-12:
        return bounding_box, NormalizedPolygon(points=normalized_points), []
    from shapely import make_valid
    from shapely.geometry import Polygon

    geometry = Polygon([(point.x, point.y) for point in normalized_points])
    if make_valid(geometry).area > 1e-12:
        raise DetectorOutputError("Detector returned an invalid self-crossing segmentation polygon")
    return bounding_box, None, [f"Detection {index} has a zero-area segmentation contour; bounding box retained."]
