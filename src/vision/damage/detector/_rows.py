"""Row-by-row conversion of raw detector output into domain detections."""

import math
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from vision.damage.detector._errors import DetectorOutputError
from vision.damage.detector._output import normalized_detection_geometry
from vision.models import VisualDamageDetection
from vision.taxonomy import VisualDamageType


@dataclass(frozen=True)
class DetectorRows:
    """Raw detector output rows, before any mapping to domain detections."""

    coordinates: list
    classes: list
    confidences: list
    polygons: list | None


def _require_predicted_class(class_id: Any, mapping: dict[int, VisualDamageType]) -> None:
    if (
        isinstance(class_id, bool)
        or not isinstance(class_id, (int, float))
        or not math.isfinite(class_id)
        or class_id != int(class_id)
        or int(class_id) not in mapping
    ):
        raise DetectorOutputError(f"Unknown or invalid predicted damage class: {class_id!r}")


def _require_predicted_confidence(confidence: Any) -> None:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise DetectorOutputError("Predicted confidence must be finite and between 0 and 1")


def _row_detection(
    index: int,
    rows: DetectorRows,
    mapping: dict[int, VisualDamageType],
    config: Any,
    warnings: list[str],
) -> VisualDamageDetection | None:
    """Validate one predicted row and keep the geometry the detector actually returned."""
    box, class_id, confidence = rows.coordinates[index], rows.classes[index], rows.confidences[index]
    _require_predicted_class(class_id, mapping)
    _require_predicted_confidence(confidence)
    polygon = rows.polygons[index] if rows.polygons is not None else None
    try:
        bounding_box, normalized, row_warnings = normalized_detection_geometry(box, polygon, index)
        warnings.extend(row_warnings)
        if rows.polygons is not None and polygon is None:
            warnings.append(f"Detection {index} has an empty binary mask; bounding box retained.")
        if confidence < config.confidence_threshold:
            return None
        return VisualDamageDetection(
            damage_type=mapping[int(class_id)],
            confidence=confidence,
            bounding_box=bounding_box,
            polygon=normalized,
            model_identifier=config.model_identifier,
            model_version=config.model_version,
        )
    except ValidationError as error:
        raise DetectorOutputError("Detector returned invalid normalized damage geometry") from error


def mapped_detections(
    rows: DetectorRows,
    mapping: dict[int, VisualDamageType],
    config: Any,
) -> tuple[list[VisualDamageDetection], list[str]]:
    detections: list[VisualDamageDetection] = []
    warnings: list[str] = []
    for index in range(len(rows.coordinates)):
        detection = _row_detection(index, rows, mapping, config, warnings)
        if detection is not None:
            detections.append(detection)
    return detections, warnings
