"""Vehicle-part baseline configuration on top of the damage training config."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import tomllib

from vision.training import REPO_ROOT, DamageTrainingConfig

DEFAULT_PARTS_CONFIG = REPO_ROOT / "configs/vision/carparts_yolo26n_seg_v1.toml"


@dataclass(frozen=True)
class VehiclePartTrainingConfig(DamageTrainingConfig):
    confidence_threshold: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if Path(self.base_model).name != "yolo26n-seg.pt":
            raise ValueError(
                "Vehicle-part V1 must initialize from independent COCO yolo26n-seg.pt; never damage best.pt"
            )
        if (
            isinstance(self.confidence_threshold, bool)
            or not math.isfinite(self.confidence_threshold)
            or not 0 <= self.confidence_threshold <= 1
        ):
            raise ValueError("Part baseline confidence threshold must be finite and between zero and one")


def load_parts_training_config(path: Path = DEFAULT_PARTS_CONFIG) -> VehiclePartTrainingConfig:
    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    unknown = set(payload) - {field.name for field in fields(VehiclePartTrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown part training keys: {sorted(unknown)}")
    payload["dataset_yaml"] = os.environ.get("PART_DATASET_YAML", payload["dataset_yaml"])
    for key in ("dataset_yaml", "output_dir"):
        value = Path(payload[key]).expanduser()
        payload[key] = (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    return VehiclePartTrainingConfig(**payload)


def require_requested_device(config: VehiclePartTrainingConfig, environment: dict[str, Any]) -> dict[str, Any]:
    """Fail explicitly if this exact interpreter cannot honor a CUDA request."""
    requested = config.device.strip().lower()
    gpu_requested = requested not in {"cpu", "mps"}
    if gpu_requested and not environment.get("cuda_available"):
        raise RuntimeError(
            "GPU training/evaluation requested but torch.cuda.is_available() is False in "
            f"{environment.get('python_executable')}; torch={environment.get('torch_version')}, "
            f"CUDA build={environment.get('cuda_runtime_version')}. "
            "Select a CUDA-enabled Python; CPU fallback is forbidden."
        )
    if gpu_requested and requested != "0":
        raise ValueError("Carparts V1 CUDA baseline requires explicit device=0")
    return environment
