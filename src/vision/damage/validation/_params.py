"""Explicit dependencies for diagnostics without importing a CV framework."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision.damage.training import DamageTrainingConfig

AP_CONFIDENCE_FLOOR = 0.001


@dataclass(frozen=True)
class ValidationParams:
    model_factory: Callable[[str], Any] | None = None
    environment_provider: Callable[[], dict[str, Any]] | None = None
    dataset_loader: Callable[[Path], dict[str, Any]] | None = None
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None
    operating_confidence: float | None = None
    image_size: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class ValidationRequest:
    config: DamageTrainingConfig
    params: ValidationParams
    output_dir: Path
    manifest: dict[str, Any]
    checkpoint: Path
    summary: dict[str, Any]
    operating_confidence: float
    image_size: int
    batch_size: int

    def payload(self) -> dict[str, Any]:
        return {
            "split": "val",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.summary["best_checkpoint_sha256"],
            "checkpoint_selection_split": "val",
            "dataset_yaml": str(self.config.dataset_yaml),
            "output_dir": str(self.output_dir),
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "device_requested": self.config.device,
            "ap_confidence_floor": AP_CONFIDENCE_FLOOR,
            "operating_confidence": self.operating_confidence,
            "validation_passes": 2,
        }
