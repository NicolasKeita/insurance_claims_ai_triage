"""Run identity, checkpoint records and the observed training outcome."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision.training._artifacts import file_sha256, verify_class_names
from vision.training._config import DamageTrainingConfig


@dataclass(frozen=True)
class PreparedRun:
    """Validated inputs shared by the damage and vehicle-part baselines."""

    manifest: dict[str, Any]
    threshold: float
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]]
    recovery: dict[str, Any] | None


@dataclass(frozen=True)
class RunIdentity:
    """Everything that identifies one run before the framework is invoked."""

    config: DamageTrainingConfig
    prepared: PreparedRun
    environment: dict[str, Any]
    provenance: dict[str, Any]
    base_checkpoint: Path
    base_checkpoint_sha256: str | None
    resume: bool
    previous_elapsed: float
    started: float
    progress: dict[str, Any]


@dataclass(frozen=True)
class ObservedTraining:
    trainer: Any
    actual_best: Path
    actual_last: Path
    epochs_completed: int | None
    metrics: dict[str, Any]


def base_checkpoint_identity(
    model: Any,
    config: DamageTrainingConfig,
    recovery: dict[str, Any] | None,
) -> tuple[Path, str | None]:
    """Resolve the pretrained starting weights and their recorded digest."""
    source = recovery["base_checkpoint"] if recovery else getattr(model, "ckpt_path", config.base_model)
    base_checkpoint = Path(source).resolve()
    if recovery:
        return base_checkpoint, recovery["base_checkpoint_sha256"]
    return base_checkpoint, file_sha256(base_checkpoint) if base_checkpoint.is_file() else None


def require_expected_checkpoints(trainer: Any, config: DamageTrainingConfig) -> tuple[Path, Path]:
    """Fail unless the selected and final checkpoints are exactly the configured artifacts."""
    expected_best = config.output_dir / "train/weights/best.pt"
    actual_best = Path(trainer.best).resolve()
    if actual_best != expected_best.resolve() or not actual_best.is_file():
        raise RuntimeError(f"Training did not produce the expected selected checkpoint: {expected_best}")
    expected_last = config.output_dir / "train/weights/last.pt"
    actual_last = Path(trainer.last).resolve()
    if actual_last != expected_last.resolve() or not actual_last.is_file():
        raise RuntimeError(f"Training did not produce the expected final checkpoint: {expected_last}")
    return actual_best, actual_last


def observe_training(identity: RunIdentity, model: Any, result: Any) -> ObservedTraining:
    """Verify the produced checkpoints and read the observed TRAIN/VAL outcome."""
    trainer = model.trainer
    actual_best, actual_last = require_expected_checkpoints(trainer, identity.config)
    verify_class_names(getattr(model, "names", None), identity.prepared.manifest["category_mapping"])
    actual_device = getattr(trainer, "device", None)
    identity.environment["device_used"] = str(actual_device) if actual_device is not None else None
    epoch = getattr(trainer, "epoch", None)
    from vision.evaluation import extract_segmentation_metrics

    metrics = extract_segmentation_metrics(
        result,
        identity.prepared.manifest["category_mapping"],
        identity.prepared.manifest["splits"]["val"],
    )
    return ObservedTraining(
        trainer=trainer,
        actual_best=actual_best,
        actual_last=actual_last,
        epochs_completed=int(epoch) + 1 if epoch is not None else None,
        metrics=metrics,
    )
