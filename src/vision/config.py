"""Central inference settings. The default is a trained, local checkpoint."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_MODEL_PATH = ROOT / "artifacts/vision/damage_segmentation_v1/train/weights/best.pt"


@dataclass(frozen=True)
class VisionSettings:
    model_path: Path = DEFAULT_MODEL_PATH
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    device: str | None = None
    max_image_bytes: int = 50 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    inference_image_size: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "model_path", Path(self.model_path).expanduser())
        threshold = self.confidence_threshold
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Damage confidence threshold must be finite and between 0 and 1")
        if self.max_image_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("Image size limits must be positive")
        if self.device is not None and not self.device.strip():
            raise ValueError("Damage inference device must be nonempty when supplied")
        if self.inference_image_size is not None and (
            isinstance(self.inference_image_size, bool)
            or not isinstance(self.inference_image_size, int)
            or self.inference_image_size < 1
        ):
            raise ValueError("Inference image size must be a positive integer")

    @classmethod
    def from_env(cls):
        return cls(
            model_path=Path(os.environ.get("DAMAGE_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            confidence_threshold=float(os.environ.get("DAMAGE_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)),
            device=os.environ.get("DAMAGE_DEVICE") or None,
        )
