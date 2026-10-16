"""Vehicle-part baseline training on the shared segmentation orchestration."""

from vision.parts_training._config import (
    DEFAULT_PARTS_CONFIG,
    VehiclePartTrainingConfig,
    load_parts_training_config,
    require_requested_device,
)
from vision.parts_training._training import PartsTrainingHooks, run_parts_training

__all__ = [
    "DEFAULT_PARTS_CONFIG",
    "PartsTrainingHooks",
    "VehiclePartTrainingConfig",
    "load_parts_training_config",
    "require_requested_device",
    "run_parts_training",
]
