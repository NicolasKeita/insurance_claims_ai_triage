"""Damage detector contract and its explicit failure modes."""

from pathlib import Path
from typing import Protocol

from vision.models import ImageDamageAssessment, VisionModelMetadata


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
