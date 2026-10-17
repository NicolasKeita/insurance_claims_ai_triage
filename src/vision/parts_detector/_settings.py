"""Vehicle-part inference settings, contract and error type."""

import os
from dataclasses import dataclass
from pathlib import Path

from vision.config import DEFAULT_CONFIDENCE_THRESHOLD, ROOT, VisionSettings
from vision.detector import DamageDetectorError
from vision.models import ImagePartAssessment, VisionModelMetadata

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


class VehiclePartDetector:
    @property
    def confidence_threshold(self) -> float: ...

    @property
    def model_metadata(self) -> VisionModelMetadata: ...

    def assess_image(self, path: Path) -> ImagePartAssessment: ...