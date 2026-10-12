"""Lazy Ultralytics adapter and a deterministic fake for dataset-free tests."""

import math
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from pydantic import ValidationError

from vision.config import VisionSettings
from vision.images import load_validated_image
from vision.models import (
    ImageDamageAssessment, NormalizedBoundingBox, NormalizedPoint,
    NormalizedPolygon, VisualDamageDetection, VisionModelMetadata,
)
from vision.taxonomy import VisualDamageType, map_category_name


class DamageDetectorError(RuntimeError):
    pass


class DetectorOutputError(DamageDetectorError):
    pass


class DamageDetector(Protocol):
    @property
    def confidence_threshold(self) -> float: ...

    @property
    def model_metadata(self) -> VisionModelMetadata: ...

    def assess_image(self, path: Path) -> ImageDamageAssessment: ...


def require_local_checkpoint(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".pt" or path.stat().st_size == 0:
        raise DamageDetectorError(
            f"Trained damage checkpoint unavailable: {path}. Prepare CarDD and run "
            "scripts/train_damage_segmentation.py, or set DAMAGE_MODEL_PATH to an existing "
            "trained damage segmentation .pt file. No weights were downloaded."
        )
    return path.resolve()


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
    if require_complete:
        if set(mapping) != set(range(len(VisualDamageType))) or set(mapping.values()) != set(VisualDamageType):
            raise DetectorOutputError("Checkpoint taxonomy must match all six supported CarDD damage types")
    return mapping


def map_ultralytics_result(
    result: Any, *, filename: str, width: int, height: int,
    confidence_threshold: float, model_identifier: str, model_version: str | None = None,
) -> ImageDamageAssessment:
    """Convert framework-like values without importing Ultralytics or torch."""
    VisionSettings(confidence_threshold=confidence_threshold)
    mapping = category_mapping_from_model_names(result.names)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ImageDamageAssessment(filename=filename, width=width, height=height)
    coordinates, classes, confidences = _list(boxes.xyxyn), _list(boxes.cls), _list(boxes.conf)
    if not len(coordinates) == len(classes) == len(confidences):
        raise DetectorOutputError("Detector box/class/confidence counts differ")
    masks = getattr(result, "masks", None)
    polygons = list(masks.xyn) if masks is not None else None
    if polygons is not None and len(polygons) != len(coordinates):
        raise DetectorOutputError("Detector mask and box counts differ")
    detections = []
    assessment_warnings = []
    for index, (box, class_id, confidence) in enumerate(zip(coordinates, classes, confidences)):
        if (isinstance(class_id, bool) or not isinstance(class_id, (int, float))
                or not math.isfinite(class_id) or class_id != int(class_id) or int(class_id) not in mapping):
            raise DetectorOutputError(f"Unknown or invalid predicted damage class: {class_id!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise DetectorOutputError("Predicted confidence must be finite and between 0 and 1")
        if len(box) != 4:
            raise DetectorOutputError("Detector box must contain four normalized coordinates")
        try:
            bounding_box = NormalizedBoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3])
            polygon = None
            if polygons is not None:
                points = _list(polygons[index])
                if len(points) < 3:
                    assessment_warnings.append(f"Detection {index} has no usable segmentation polygon; bounding box retained.")
                else:
                    if any(len(point) != 2 for point in points):
                        raise DetectorOutputError("Each polygon point must contain x and y")
                    polygon = NormalizedPolygon(points=[NormalizedPoint(x=point[0], y=point[1]) for point in points])
            if confidence >= confidence_threshold:
                detections.append(VisualDamageDetection(
                    damage_type=mapping[int(class_id)], confidence=confidence,
                    bounding_box=bounding_box, polygon=polygon,
                    model_identifier=model_identifier, model_version=model_version,
                ))
        except ValidationError as error:
            raise DetectorOutputError("Detector returned invalid normalized damage geometry") from error
    if masks is None and detections:
        assessment_warnings.append("Segmentation masks unavailable for these detections; bounding boxes retained.")
    return ImageDamageAssessment(
        filename=filename, width=width, height=height,
        detections=detections, warnings=assessment_warnings,
    )


class UltralyticsDamageDetector:
    def __init__(self, settings: VisionSettings | None = None, *, model_factory: Callable | None = None):
        self.settings = settings or VisionSettings.from_env()
        self._model_factory = model_factory
        self._model = None
        self._load_lock = Lock()

    @property
    def confidence_threshold(self) -> float:
        return self.settings.confidence_threshold

    @property
    def model_metadata(self) -> VisionModelMetadata:
        try:
            library_version = version("ultralytics")
        except PackageNotFoundError:
            library_version = None
        return VisionModelMetadata(
            identifier=str(self.settings.model_path.resolve()), framework="ultralytics", framework_version=library_version,
        )

    def _load_model(self):
        with self._load_lock:
            if self._model is not None:
                return self._model
            checkpoint = require_local_checkpoint(self.settings.model_path)
            factory = self._model_factory
            if factory is None:
                try:
                    from ultralytics import YOLO
                except ImportError as error:
                    raise DamageDetectorError("Install the optional vision extra: python -m pip install -e '.[vision]'") from error
                factory = YOLO
            try:
                model = factory(str(checkpoint))
            except Exception as error:
                raise DamageDetectorError(f"Could not load local damage checkpoint: {checkpoint}") from error
            if getattr(model, "task", None) != "segment":
                raise DamageDetectorError("Damage checkpoint must use the Ultralytics segment task")
            category_mapping_from_model_names(model.names, require_complete=True)
            self._model = model
            return model

    def assess_image(self, path: Path) -> ImageDamageAssessment:
        path = Path(path)
        # Decode first: invalid images cannot trigger a model import or load.
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
            metadata = self.model_metadata
            assessment = map_ultralytics_result(
                result, filename=path.name, width=image.width, height=image.height,
                confidence_threshold=self.confidence_threshold,
                model_identifier=metadata.identifier, model_version=metadata.version,
            )
            return assessment.model_copy(update={"asset_reference": str(path)})
        finally:
            image.close()


class FakeDamageDetector:
    """Explicit synthetic observations for tests; never a real damage model."""

    def __init__(
        self, detections_by_filename: Mapping[str, list[VisualDamageDetection]] | None = None,
        *, confidence_threshold: float = 0.25,
    ):
        self.settings = VisionSettings(confidence_threshold=confidence_threshold)
        self.detections_by_filename = dict(detections_by_filename or {})
        self.calls: list[Path] = []

    @property
    def confidence_threshold(self) -> float:
        return self.settings.confidence_threshold

    @property
    def model_metadata(self) -> VisionModelMetadata:
        return VisionModelMetadata(identifier="fake-damage-detector", framework="fake", version="test-only")

    def assess_image(self, path: Path) -> ImageDamageAssessment:
        path = Path(path)
        image = load_validated_image(path, self.settings)
        try:
            self.calls.append(path)
            return ImageDamageAssessment(
                filename=path.name, asset_reference=str(path), width=image.width, height=image.height,
                detections=[item for item in self.detections_by_filename.get(path.name, []) if item.confidence >= self.confidence_threshold],
            )
        finally:
            image.close()
