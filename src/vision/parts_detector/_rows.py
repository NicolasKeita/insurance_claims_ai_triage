"""Row-by-row conversion of raw part detector output into domain detections."""

import math
from typing import Any

from pydantic import ValidationError

from vision.detector import DetectorOutputError, normalized_detection_geometry
from vision.detector._rows import DetectorRows
from vision.geometry import PolygonGeometryError, validated_polygon_geometry
from vision.models import VehiclePartDetection
from vision.taxonomy import VehiclePartType


def _require_part_class(class_id: Any, mapping: dict[int, VehiclePartType]) -> None:
    if (
        isinstance(class_id, bool)
        or not isinstance(class_id, (int, float))
        or not math.isfinite(class_id)
        or class_id != int(class_id)
        or int(class_id) not in mapping
    ):
        raise DetectorOutputError(f"Unknown or invalid predicted vehicle part class: {class_id!r}")


def _require_confidence(confidence: Any) -> None:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise DetectorOutputError("Predicted confidence must be finite and between 0 and 1")


def _part_row(
    index: int,
    rows: DetectorRows,
    mapping: dict[int, VehiclePartType],
    config: Any,
    warnings: list[str],
) -> VehiclePartDetection | None:
    """Validate one predicted row, including the reusable CarDD polygon geometry."""
    box, class_id, confidence = rows.coordinates[index], rows.classes[index], rows.confidences[index]
    _require_part_class(class_id, mapping)
    _require_confidence(confidence)
    polygon = rows.polygons[index] if rows.polygons is not None else None
    try:
        bounding_box, normalized, geometry_warnings = normalized_detection_geometry(box, polygon, index)
        if normalized is not None:
            validated_polygon_geometry(normalized)
        warnings.extend(geometry_warnings)
        if rows.polygons is not None and polygon is None:
            warnings.append(f"Detection {index} has an empty binary mask; bounding box retained.")
        if confidence < config.confidence_threshold:
            return None
        return VehiclePartDetection(
            part_type=mapping[int(class_id)],
            confidence=confidence,
            bounding_box=bounding_box,
            polygon=normalized,
            model_identifier=config.model_metadata.identifier,
            model_version=config.model_metadata.version,
        )
    except (ValidationError, PolygonGeometryError) as error:
        raise DetectorOutputError("Detector returned invalid normalized vehicle part geometry") from error


def part_detections(
    rows: DetectorRows,
    mapping: dict[int, VehiclePartType],
    config: Any,
) -> tuple[list[VehiclePartDetection], list[str]]:
    detections: list[VehiclePartDetection] = []
    warnings: list[str] = []
    for index in range(len(rows.coordinates)):
        detection = _part_row(index, rows, mapping, config, warnings)
        if detection is not None:
            detections.append(detection)
    return detections, warnings
