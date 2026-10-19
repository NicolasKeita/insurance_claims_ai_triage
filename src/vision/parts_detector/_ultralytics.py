"""Independent lazy Carparts-Seg adapter; no framework is imported before use."""

from collections.abc import Callable
from pathlib import Path
from threading import Lock

from vision.detector import DetectorOutputError, local_model_metadata
from vision.images import load_validated_image
from vision.models import ImagePartAssessment, VisionModelMetadata
from vision.parts_detector._mapping import PartMappingConfig, map_ultralytics_part_result, part_mapping_from_model_names
from vision.parts_detector._settings import VehiclePartDetectorError, VehiclePartSettings


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
                    raise VehiclePartDetectorError(
                        "Install the optional vision extra: python -m pip install -e '.[vision]'"
                    ) from error
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
            options = {"source": image, "conf": self.confidence_threshold, "verbose": False, "retina_masks": True}
            if self.settings.device is not None:
                options["device"] = self.settings.device
            results = model.predict(**options)
            if len(results) != 1:
                raise DetectorOutputError("One input image must produce one result")
            result = results[0]
            if getattr(result, "orig_shape", (image.height, image.width)) != (image.height, image.width):
                raise DetectorOutputError("Detector result dimensions differ from the EXIF-oriented image")
            assessment = map_ultralytics_part_result(
                result,
                config=PartMappingConfig(
                    filename=path.name,
                    width=image.width,
                    height=image.height,
                    confidence_threshold=self.confidence_threshold,
                    model_metadata=self.model_metadata,
                ),
            )
            return assessment.model_copy(update={"asset_reference": str(path)})
        finally:
            image.close()
