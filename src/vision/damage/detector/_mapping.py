"""Model class-name mapping and result-to-assessment conversion."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vision.config import VisionSettings
from vision.damage.detector._errors import DetectorOutputError
from vision.damage.detector._output import extract_ultralytics_rows
from vision.damage.detector._rows import DetectorRows, mapped_detections
from vision.models import ImageDamageAssessment
from vision.taxonomy import VisualDamageType, map_category_name


def category_mapping_from_model_names(names: Any, *, require_complete: bool = False) -> dict[int, VisualDamageType]:
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, Mapping) or not names:
        raise DetectorOutputError("Model must expose its damage class names")
    mapping = {}
    try:
        for class_id, name in names.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
                raise ValueError("Model class IDs must be nonnegative integers")
            mapping[class_id] = map_category_name(name)
    except ValueError as error:
        raise DetectorOutputError(str(error)) from error
    if len(set(mapping.values())) != len(mapping):
        raise DetectorOutputError("Model maps multiple class IDs to the same damage type")
    if require_complete and (
        set(mapping) != set(range(len(VisualDamageType))) or set(mapping.values()) != set(VisualDamageType)
    ):
        raise DetectorOutputError("Checkpoint taxonomy must match all six supported CarDD damage types")
    return mapping


@dataclass
class DamageMappingConfig:
    """Image dimensions, confidence threshold and damage model provenance."""

    filename: str
    width: int
    height: int
    confidence_threshold: float
    model_identifier: str
    model_version: str | None = None


def map_ultralytics_result(result: Any, *, config: DamageMappingConfig) -> ImageDamageAssessment:
    """Convert framework-like values without importing Ultralytics or torch."""
    VisionSettings(confidence_threshold=config.confidence_threshold)
    mapping = category_mapping_from_model_names(result.names)
    coordinates, classes, confidences, polygons = extract_ultralytics_rows(
        result, width=config.width, height=config.height
    )
    rows = DetectorRows(coordinates, classes, confidences, polygons)
    detections, warnings = mapped_detections(rows, mapping, config)
    if polygons is None and detections:
        warnings.append("Segmentation masks unavailable for these detections; bounding boxes retained.")
    return ImageDamageAssessment(
        filename=config.filename,
        width=config.width,
        height=config.height,
        detections=detections,
        warnings=warnings,
    )
