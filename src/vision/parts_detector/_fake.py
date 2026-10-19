"""Synthetic part observations for tests, never pretrained model output."""

from collections.abc import Mapping
from pathlib import Path

from vision.config import DEFAULT_CONFIDENCE_THRESHOLD
from vision.images import load_validated_image
from vision.models import ImagePartAssessment, VehiclePartDetection, VisionModelMetadata
from vision.parts_detector._settings import VehiclePartSettings


class FakeVehiclePartDetector:
    def __init__(
        self,
        detections_by_filename: Mapping[str, list[VehiclePartDetection]] | None = None,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
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
                filename=path.name,
                asset_reference=str(path),
                width=image.width,
                height=image.height,
                detections=[
                    item
                    for item in self.detections_by_filename.get(path.name, [])
                    if item.confidence >= self.confidence_threshold
                ],
                model_metadata=self.model_metadata,
                confidence_threshold=self.confidence_threshold,
            )
        finally:
            image.close()
