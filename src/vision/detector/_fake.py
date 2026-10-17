"""Explicit synthetic observations for tests; never a real damage model."""

from collections.abc import Mapping
from pathlib import Path

from vision.config import VisionSettings
from vision.images import load_validated_image
from vision.models import ImageDamageAssessment, VisionModelMetadata, VisualDamageDetection


class FakeDamageDetector:
    def __init__(
        self,
        detections_by_filename: Mapping[str, list[VisualDamageDetection]] | None = None,
        *,
        confidence_threshold: float = 0.25,
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
                filename=path.name,
                asset_reference=str(path),
                width=image.width,
                height=image.height,
                detections=[
                    item
                    for item in self.detections_by_filename.get(path.name, [])
                    if item.confidence >= self.confidence_threshold
                ],
            )
        finally:
            image.close()
