"""Lazy Ultralytics damage adapter; no framework is imported before use."""

from collections.abc import Callable
from pathlib import Path
from threading import Lock

from vision.config import VisionSettings
from vision.detector._checkpoint import local_model_metadata, require_local_checkpoint
from vision.detector._errors import DamageDetectorError, DetectorOutputError
from vision.detector._mapping import DamageMappingConfig, category_mapping_from_model_names, map_ultralytics_result
from vision.images import load_validated_image
from vision.models import ImageDamageAssessment, VisionModelMetadata


class UltralyticsDamageDetector:
    def __init__(self, settings: VisionSettings | None = None, *, model_factory: Callable | None = None):
        self.settings = settings or VisionSettings.from_env()
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
            checkpoint = require_local_checkpoint(self.settings.model_path)
            factory = self._model_factory
            if factory is None:
                try:
                    from ultralytics import YOLO
                except ImportError as error:
                    raise DamageDetectorError(
                        "Install the optional vision extra: python -m pip install -e '.[vision]'"
                    ) from error
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
        """Decode first: invalid images cannot trigger a model import or load."""
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
            metadata = self.model_metadata
            assessment = map_ultralytics_result(
                result,
                config=DamageMappingConfig(
                    filename=path.name,
                    width=image.width,
                    height=image.height,
                    confidence_threshold=self.confidence_threshold,
                    model_identifier=metadata.identifier,
                    model_version=metadata.version,
                ),
            )
            return assessment.model_copy(update={"asset_reference": str(path)})
        finally:
            image.close()
