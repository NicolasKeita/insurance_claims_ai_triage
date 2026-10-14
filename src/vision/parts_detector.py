"""Independent lazy Carparts-Seg adapter, sharing step29 image/geometry helpers."""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from pydantic import ValidationError

from vision.config import DEFAULT_CONFIDENCE_THRESHOLD, ROOT, VisionSettings
from vision.detector import (
    DamageDetectorError, DetectorOutputError, extract_ultralytics_rows,
    local_model_metadata, normalized_detection_geometry,
)
from vision.images import load_validated_image
from vision.geometry import PolygonGeometryError, validated_polygon_geometry
from vision.models import ImagePartAssessment, VehiclePartDetection, VisionModelMetadata
from vision.taxonomy import CARPARTS_CLASS_TO_PART, VehiclePartType, map_vehicle_part_category_name

DEFAULT_PART_MODEL_PATH = ROOT / "artifacts/vision/vehicle_parts_v1/train/weights/best.pt"


@dataclass(frozen=True)
class VehiclePartSettings(VisionSettings):
    model_path: Path = DEFAULT_PART_MODEL_PATH

    @classmethod
    def from_env(cls):
        return cls(
            model_path=Path(os.environ.get("PART_MODEL_PATH", str(DEFAULT_PART_MODEL_PATH))),
            confidence_threshold=float(os.environ.get("PART_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)),
            device=os.environ.get("PART_DEVICE") or None,
        )


class VehiclePartDetectorError(DamageDetectorError):
    pass


class VehiclePartDetector(Protocol):
    @property
    def confidence_threshold(self) -> float: ...

    @property
    def model_metadata(self) -> VisionModelMetadata: ...

    def assess_image(self, path: Path) -> ImagePartAssessment: ...


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


def map_ultralytics_part_result(
    result: Any, *, filename: str, width: int, height: int,
    confidence_threshold: float, model_metadata: VisionModelMetadata,
) -> ImagePartAssessment:
    """No torch or Ultralytics import is needed to map framework-like output."""
    VehiclePartSettings(confidence_threshold=confidence_threshold)
    mapping = part_mapping_from_model_names(result.names)
    coordinates, classes, confidences, polygons = extract_ultralytics_rows(result)
    detections, warnings = [], []
    for index, (box, class_id, confidence) in enumerate(zip(coordinates, classes, confidences)):
        if (isinstance(class_id, bool) or not isinstance(class_id, (int, float))
                or not math.isfinite(class_id) or class_id != int(class_id) or int(class_id) not in mapping):
            raise DetectorOutputError(f"Unknown or invalid predicted vehicle part class: {class_id!r}")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise DetectorOutputError("Predicted confidence must be finite and between 0 and 1")
        try:
            bounding_box, polygon, geometry_warnings = normalized_detection_geometry(
                box, polygons[index] if polygons is not None else None, index,
            )
            if polygon is not None:
                validated_polygon_geometry(polygon)
            warnings.extend(geometry_warnings)
            if confidence >= confidence_threshold:
                detections.append(VehiclePartDetection(
                    part_type=mapping[int(class_id)], confidence=confidence,
                    bounding_box=bounding_box, polygon=polygon,
                    model_identifier=model_metadata.identifier, model_version=model_metadata.version,
                ))
        except (ValidationError, PolygonGeometryError) as error:
            raise DetectorOutputError("Detector returned invalid normalized vehicle part geometry") from error
    if polygons is None and detections:
        warnings.append("Segmentation masks unavailable for these detections; bounding boxes retained.")
    return ImagePartAssessment(
        filename=filename, width=width, height=height, detections=detections,
        warnings=warnings, model_metadata=model_metadata, confidence_threshold=confidence_threshold,
    )


class UltralyticsVehiclePartDetector:
    def __init__(self, settings: VehiclePartSettings | None = None, *, model_factory: Callable | None = None):
        self.settings = settings or VehiclePartSettings.from_env()
        self._model_factory = model_factory
        self._model = None
        self._metadata = None
        self._load_lock = Lock()

    @property
    def confidence_threshold(self) -> float:
        return self.settings.confidence_threshold

    @property
    def model_metadata(self) -> VisionModelMetadata:
        if self._metadata is None or self._metadata.version is None:
            self._metadata = local_model_metadata(self.settings.model_path)
        return self._metadata

    def _load_model(self):
        with self._load_lock:
            if self._model is not None:
                return self._model
            checkpoint = self.settings.model_path
            if not checkpoint.is_file() or checkpoint.suffix.casefold() != ".pt" or checkpoint.stat().st_size == 0:
                raise VehiclePartDetectorError(
                    f"Trained vehicle part checkpoint unavailable: {checkpoint}. Run "
                    "scripts/train_vehicle_part_segmentation.py or set PART_MODEL_PATH to a trained "
                    "local Carparts-Seg segmentation checkpoint. No weights were downloaded."
                )
            factory = self._model_factory
            if factory is None:
                try:
                    from ultralytics import YOLO
                except ImportError as error:
                    raise VehiclePartDetectorError("Install the optional vision extra: python -m pip install -e '.[vision]'") from error
                factory = YOLO
            try:
                model = factory(str(checkpoint.resolve()))
            except Exception as error:
                raise VehiclePartDetectorError(f"Could not load local vehicle part checkpoint: {checkpoint}") from error
            if getattr(model, "task", None) != "segment":
                raise VehiclePartDetectorError("Vehicle part checkpoint must use the Ultralytics segment task")
            part_mapping_from_model_names(model.names, require_complete=True)
            self._model = model
            return model

    def assess_image(self, path: Path) -> ImagePartAssessment:
        path = Path(path)
        image = load_validated_image(path, self.settings)
        try:
            model = self._load_model()
            options = {"source": image, "conf": self.confidence_threshold, "verbose": False}
            if self.settings.device is not None:
                options["device"] = self.settings.device
            results = model.predict(**options)
            if len(results) != 1:
                raise DetectorOutputError("One input image must produce one result")
            result = results[0]
            if getattr(result, "orig_shape", (image.height, image.width)) != (image.height, image.width):
                raise DetectorOutputError("Detector result dimensions differ from the EXIF-oriented image")
            assessment = map_ultralytics_part_result(
                result, filename=path.name, width=image.width, height=image.height,
                confidence_threshold=self.confidence_threshold, model_metadata=self.model_metadata,
            )
            return assessment.model_copy(update={"asset_reference": str(path)})
        finally:
            image.close()


class FakeVehiclePartDetector:
    """Synthetic part observations for tests, never pretrained model output."""

    def __init__(
        self, detections_by_filename: Mapping[str, list[VehiclePartDetection]] | None = None,
        *, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.settings = VehiclePartSettings(confidence_threshold=confidence_threshold)
        self.detections_by_filename = dict(detections_by_filename or {})
        self.calls: list[Path] = []

    @property
    def confidence_threshold(self) -> float:
        return self.settings.confidence_threshold

    @property
    def model_metadata(self) -> VisionModelMetadata:
        return VisionModelMetadata(identifier="fake-vehicle-part-detector", framework="fake", version="test-only")

    def assess_image(self, path: Path) -> ImagePartAssessment:
        path = Path(path)
        image = load_validated_image(path, self.settings)
        try:
            self.calls.append(path)
            return ImagePartAssessment(
                filename=path.name, asset_reference=str(path), width=image.width, height=image.height,
                detections=[item for item in self.detections_by_filename.get(path.name, []) if item.confidence >= self.confidence_threshold],
                model_metadata=self.model_metadata, confidence_threshold=self.confidence_threshold,
            )
        finally:
            image.close()
