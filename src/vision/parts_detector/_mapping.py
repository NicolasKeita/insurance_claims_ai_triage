"""Carparts-Seg class-name mapping and result-to-assessment conversion."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vision.detector import DetectorOutputError, extract_ultralytics_rows
from vision.detector._rows import DetectorRows
from vision.models import ImagePartAssessment, VisionModelMetadata
from vision.parts_detector._rows import part_detections
from vision.parts_detector._settings import VehiclePartSettings
from vision.taxonomy import CARPARTS_CLASS_TO_PART, VehiclePartType, map_vehicle_part_category_name


def part_mapping_from_model_names(names: Any, *, require_complete: bool = False) -> dict[int, VehiclePartType]:
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, Mapping) or not names:
        raise DetectorOutputError("Model must expose its Carparts-Seg class names")
    mapping = {}
    try:
        for class_id, name in names.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
                raise ValueError("Model class IDs must be nonnegative integers")
            mapping[class_id] = map_vehicle_part_category_name(name)
    except ValueError as error:
        raise DetectorOutputError(str(error)) from error
    if len(set(mapping.values())) != len(mapping):
        raise DetectorOutputError("Model maps multiple class IDs to the same vehicle part")
    if require_complete and (
        set(mapping) != set(range(len(CARPARTS_CLASS_TO_PART)))
        or set(mapping.values()) != set(CARPARTS_CLASS_TO_PART.values())
    ):
        raise DetectorOutputError("Checkpoint taxonomy must match all 23 supported Carparts-Seg classes")
    return mapping


@dataclass
class PartMappingConfig:
    """Image dimensions, confidence threshold and vehicle-part model provenance."""

    filename: str
    width: int
    height: int
    confidence_threshold: float
    model_metadata: VisionModelMetadata


def map_ultralytics_part_result(result: Any, *, config: PartMappingConfig) -> ImagePartAssessment:
    """No torch or Ultralytics import is needed to map framework-like output."""
    VehiclePartSettings(confidence_threshold=config.confidence_threshold)
    mapping = part_mapping_from_model_names(result.names)
    coordinates, classes, confidences, polygons = extract_ultralytics_rows(
        result, width=config.width, height=config.height
    )
    rows = DetectorRows(coordinates, classes, confidences, polygons)
    detections, warnings = part_detections(rows, mapping, config)
    if polygons is None and detections:
        warnings.append("Segmentation masks unavailable for these detections; bounding boxes retained.")
    return ImagePartAssessment(
        filename=config.filename,
        width=config.width,
        height=config.height,
        detections=detections,
        warnings=warnings,
        model_metadata=config.model_metadata,
        confidence_threshold=config.confidence_threshold,
    )