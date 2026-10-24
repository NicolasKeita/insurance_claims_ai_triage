"""Versioned real-evaluation configuration and environment overrides."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import tomllib

from evaluation import E2E_BENCHMARK_VERSION

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/evaluation/e2e_v2.toml"
SWITCH_KEYS = (
    "include_vision",
    "include_agent",
    "include_rag",
    "include_similar",
    "fail_fast",
    "dependency_unavailable",
    "timeout_seconds",
)
CHECKPOINT_KEYS = ("triage_checkpoint", "anomaly_checkpoint", "damage_checkpoint", "part_checkpoint")
ENVIRONMENT_CHECKPOINTS = (("damage_checkpoint", "DAMAGE_MODEL_PATH"), ("part_checkpoint", "PART_MODEL_PATH"))


@dataclass(frozen=True)
class RealEvaluationConfig:
    include_vision: bool = True
    include_agent: bool = True
    include_rag: bool = True
    include_similar: bool = True
    fail_fast: bool = True
    dependency_unavailable: str = "SKIP"
    timeout_seconds: float = 2.0
    database_url: str | None = None
    qdrant_url: str | None = None
    llm_model: str | None = None
    triage_checkpoint: Path = ROOT / "artifacts/triage_v1/logistic_regression.joblib"
    anomaly_checkpoint: Path = ROOT / "artifacts/anomaly_v1/anomaly_detector.joblib"
    damage_checkpoint: Path = ROOT / "artifacts/vision/damage_segmentation_v1/train/weights/best.pt"
    part_checkpoint: Path = ROOT / "artifacts/vision/vehicle_parts_v1/train/weights/best.pt"
    vision_device: str | None = None
    vision_image_size: int = 640
    vision_max_temperature_c: float = 80.0
    source_summary: Path = ROOT / "artifacts/vision/damage_segmentation_v1/test_smoke/smoke_summary.json"
    training_summary: Path = ROOT / "artifacts/vision/damage_segmentation_v1/training_summary.json"
    diagnostic_max_images: int = 2
    synthetic_declared_damage: tuple[str, ...] = ("FRONT_BUMPER",)
    synthetic_quoted_damage: tuple[str, ...] = ()
    fixture_dir: Path = ROOT / "artifacts/evaluation/e2e_v2/fixtures"
    isolated_demo_stack: bool = False

    def __post_init__(self):
        if self.dependency_unavailable not in {"SKIP", "FAIL"}:
            raise ValueError("dependency_unavailable must be SKIP or FAIL")
        if not 0 < self.timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be positive and at most 30")
        if self.diagnostic_max_images < 1:
            raise ValueError("diagnostic_max_images must be positive")
        if (
            isinstance(self.vision_image_size, bool)
            or not isinstance(self.vision_image_size, int)
            or not 32 <= self.vision_image_size <= 640
            or self.vision_image_size % 32
        ):
            raise ValueError("vision_image_size must be a multiple of 32 between 32 and 640")
        if (
            isinstance(self.vision_max_temperature_c, bool)
            or not math.isfinite(self.vision_max_temperature_c)
            or self.vision_max_temperature_c <= 0
        ):
            raise ValueError("vision_max_temperature_c must be finite and positive")

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_CONFIG, **overrides):
        """Current project environment settings take precedence over versioned defaults."""
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("benchmark_version") != E2E_BENCHMARK_VERSION:
            raise ValueError("Unsupported real evaluation benchmark_version")
        values = {key: raw[key] for key in SWITCH_KEYS if key in raw}
        values.update(checkpoint_paths(raw.get("models", {})))
        values.update(vision_diagnostic_paths(raw.get("vision_diagnostic", {})))
        safety = raw.get("vision_safety", {})
        values.update({key: safety[key] for key in ("vision_image_size", "vision_max_temperature_c") if key in safety})
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


def checkpoint_paths(models):
    """Resolve versioned checkpoint paths, then explicit environment overrides."""
    values = {}
    for key in CHECKPOINT_KEYS:
        if key in models:
            model_path = Path(models[key]).expanduser()
            values[key] = model_path if model_path.is_absolute() else ROOT / model_path
    for key, variable in ENVIRONMENT_CHECKPOINTS:
        if os.environ.get(variable):
            values[key] = Path(os.environ[variable]).expanduser()
    values["vision_device"] = os.environ.get("DAMAGE_DEVICE") or models.get("vision_device")
    return values


def vision_diagnostic_paths(visual):
    """Resolve the standalone external-image diagnostic settings."""
    values = {}
    for key in ("source_summary", "training_summary"):
        if key in visual:
            source = Path(visual[key]).expanduser()
            values[key] = source if source.is_absolute() else ROOT / source
    values["diagnostic_max_images"] = int(visual.get("max_images", 2))
    values["synthetic_declared_damage"] = tuple(visual.get("synthetic_declared_damage", ["FRONT_BUMPER"]))
    values["synthetic_quoted_damage"] = tuple(visual.get("synthetic_quoted_damage", []))
    return values
