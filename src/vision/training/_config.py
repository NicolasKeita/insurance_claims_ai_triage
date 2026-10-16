"""Frozen damage-training configuration and its injected collaborators."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from vision.config import ROOT

REPO_ROOT = ROOT
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs/vision/cardd_hf_yolo26n_seg_v1.toml"


@dataclass(frozen=True)
class DamageTrainingConfig:
    version: str
    epochs: int
    imgsz: int
    batch: int
    patience: int
    seed: int
    device: str
    workers: int
    optimizer: str
    base_model: str
    dataset_yaml: Path
    output_dir: Path
    experiment_name: str
    track_mlflow: bool = False
    log_checkpoint: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "imgsz", "batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("patience", "seed", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.device or not self.version or not self.optimizer:
            raise ValueError("version, device and optimizer must be explicit")
        if Path(self.base_model).suffix.lower() != ".pt":
            raise ValueError("DAMAGE_BASE_MODEL must be pretrained .pt weights, never a scratch YAML")
        for name in ("track_mlflow", "log_checkpoint"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_yaml"] = str(self.dataset_yaml)
        payload["output_dir"] = str(self.output_dir)
        return payload


def load_training_config(path: Path = DEFAULT_TRAINING_CONFIG) -> DamageTrainingConfig:
    """Resolve configured relative paths from the repository, regardless of cwd."""
    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    unknown = set(payload) - {field.name for field in fields(DamageTrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    payload["base_model"] = os.environ.get("DAMAGE_BASE_MODEL", payload["base_model"])
    payload["dataset_yaml"] = os.environ.get("DAMAGE_DATASET_YAML", payload["dataset_yaml"])
    for key in ("dataset_yaml", "output_dir"):
        value = Path(payload[key]).expanduser()
        payload[key] = (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    return DamageTrainingConfig(**payload)


HookBundle = TypeVar("HookBundle")


@dataclass(frozen=True)
class TrainingHooks:
    """Injected collaborators; only the model and environment are mandatory."""

    model_factory: Callable[[str], Any]
    environment_provider: Callable[[], dict[str, Any]]
    dataset_loader: Callable[[Path], dict[str, Any]] | None = None
    resume: bool = False
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None
    confidence_threshold: float | None = None
    tracking_factory: Callable[..., Any] | None = None


def build_hooks(hooks: HookBundle | None, overrides: dict[str, Any], factory: type[HookBundle]) -> HookBundle:
    """Accept an explicit hooks object or the equivalent legacy keyword hooks."""
    if hooks is None:
        return factory(**overrides)
    if overrides:
        raise TypeError("Hook keywords cannot be combined with an explicit hooks object")
    return hooks