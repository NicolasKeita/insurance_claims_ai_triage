"""Ultralytics callbacks that keep the resumable run records honest."""

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision.training._artifacts import file_sha256, write_json
from vision.training._config import DamageTrainingConfig
from vision.training._records import RunIdentity


def resume_stopper_restorer(
    config: DamageTrainingConfig,
    recovery: dict[str, Any],
    saved_checkpoint: dict[str, Any],
) -> Callable[[Any], None]:
    """Rebuild the stopper state of the interrupted run before resume continues."""

    def restore(trainer: Any) -> None:
        if trainer.start_epoch != recovery["epochs_completed"] or trainer.epochs != config.epochs:
            raise RuntimeError("Framework resume did not preserve the expected start epoch and total epoch budget")
        saved_stopper = saved_checkpoint.get("stopper") or {}
        trainer.best_fitness = saved_checkpoint["best_fitness"]
        trainer.stopper.best_fitness = saved_stopper.get("best_fitness", saved_checkpoint["best_fitness"])
        trainer.stopper.best_epoch = saved_stopper.get("best_epoch", recovery["best_epoch"])
        trainer.stopper.possible_stop = (
            (recovery["epochs_completed"] - trainer.stopper.best_epoch >= config.patience - 1)
            if config.patience
            else False
        )

    return restore


def checkpoint_progress_recorder(identity: RunIdentity) -> Callable[[Any], None]:
    """Record actual best.pt writes, including fitness ties, for a resumable run."""

    def record(trainer: Any) -> None:
        epoch_completed = int(trainer.epoch) + 1
        if trainer.fitness == trainer.best_fitness:
            identity.progress["best_epoch"] = epoch_completed
        identity.progress.update(
            {
                "status": "running",
                "epochs_completed": epoch_completed,
                "checkpoint_selection_split": "val",
                "fitness": float(trainer.fitness),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(identity.config.output_dir / "training_progress.json", identity.progress)
        recovery = identity.prepared.recovery
        write_json(
            identity.config.output_dir / "recovery_training_metadata.json",
            {
                "config": identity.config.payload(),
                "dataset": identity.provenance,
                "confidence_threshold": identity.prepared.threshold,
                "checkpoint_selection_split": "val",
                "base_checkpoint": str(identity.base_checkpoint),
                "base_checkpoint_sha256": identity.base_checkpoint_sha256,
                "best_checkpoint_sha256": file_sha256(Path(trainer.best)),
                "last_checkpoint_sha256": file_sha256(Path(trainer.last)),
                "best_epoch": identity.progress["best_epoch"],
                "epochs_completed": epoch_completed,
                "elapsed_seconds": identity.previous_elapsed + time.perf_counter() - identity.started,
                "resume_count": (recovery.get("resume_count", 0) if recovery else 0) + int(identity.resume),
            },
        )

    return record
