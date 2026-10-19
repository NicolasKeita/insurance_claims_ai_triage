"""The immutable metadata-only run summary and its publication."""

import time
from datetime import datetime, timezone
from typing import Any

from vision.training._artifacts import file_sha256, write_json
from vision.training._records import ObservedTraining, RunIdentity, observe_training


def _checkpoint_summary(identity: RunIdentity, observed: ObservedTraining, best_epoch: int) -> dict[str, Any]:
    trainer = observed.trainer
    return {
        "best_checkpoint": "train/weights/best.pt",
        "best_checkpoint_sha256": file_sha256(observed.actual_best),
        "last_checkpoint": "train/weights/last.pt",
        "last_checkpoint_sha256": file_sha256(observed.actual_last),
        "best_epoch": best_epoch,
        "best_epoch_source": "on_model_save: actual best.pt write, including fitness ties",
        "early_stopping_best_epoch": getattr(getattr(trainer, "stopper", None), "best_epoch", None),
    }


def _epoch_summary(identity: RunIdentity, observed: ObservedTraining, early_stopped: bool | None) -> dict[str, Any]:
    trainer = observed.trainer
    recovery = identity.prepared.recovery
    return {
        "epochs_completed": observed.epochs_completed,
        "effective_batch_size": getattr(trainer, "batch_size", identity.config.batch),
        "best_validation_fitness": float(trainer.best_fitness)
        if getattr(trainer, "best_fitness", None) is not None
        else None,
        "stopped_before_requested_epochs": early_stopped,
        "early_stopping_triggered": early_stopped,
        "elapsed_seconds": identity.previous_elapsed + time.perf_counter() - identity.started,
        "elapsed_seconds_before_resume": identity.previous_elapsed,
        "elapsed_seconds_this_execution": time.perf_counter() - identity.started,
        "elapsed_seconds_method": "Sum of checkpoint-recorded prior duration and this execution; "
        "excludes interruption downtime. Native results.csv time restarts for resumed segments.",
        "resumed": identity.resume,
        "resume_from_epoch": recovery["epochs_completed"] if recovery else None,
        "resume_input_checkpoint_sha256": recovery["last_checkpoint_sha256"] if recovery else None,
        "validation_metrics": observed.metrics,
        "reproducibility": "seeded / reproducible configuration; bitwise GPU determinism is not guaranteed",
        "registry": "NOT EXECUTED — custom vision registry packaging deferred; no production alias",
        "training_artifacts": {
            "results_csv": "train/results.csv",
            "results_png": "train/results.png",
            "args_yaml": "train/args.yaml",
        },
    }


def build_summary_payload(identity: RunIdentity, observed: ObservedTraining) -> dict[str, Any]:
    """Assemble the summary of a TRAIN/VAL run; TEST is never used for selection."""
    best_epoch = identity.progress["best_epoch"]
    if best_epoch is None:
        raise RuntimeError("Training did not record the epoch that wrote the selected best.pt checkpoint")
    early_stopped = (
        observed.epochs_completed < identity.config.epochs if observed.epochs_completed is not None else None
    )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "task": "segment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": identity.config.payload(),
        "environment": identity.environment,
        "dataset": identity.provenance,
        "confidence_threshold": identity.prepared.threshold,
        "checkpoint_selection_split": "val",
        "training_optimization_split": "train",
        "training_validation_split": "val",
        "test_used_during_training": False,
        "base_checkpoint": str(identity.base_checkpoint),
        "base_checkpoint_sha256": identity.base_checkpoint_sha256,
    }
    summary.update(_checkpoint_summary(identity, observed, best_epoch))
    summary.update(_epoch_summary(identity, observed, early_stopped))
    return summary


def publish_summary(
    identity: RunIdentity,
    observed: ObservedTraining,
    summary: dict[str, Any],
    tracking: Any,
) -> None:
    """Write the summary, mirror it to tracking and close the progress record."""
    summary_path = identity.config.output_dir / "training_summary.json"
    write_json(summary_path, summary)
    if tracking is not None:
        tracking.log_training(summary, summary_path, observed.actual_best)
        summary["mlflow_run_id"] = tracking.run_id
        write_json(summary_path, summary)
    identity.progress.update({"status": "completed", "epochs_completed": observed.epochs_completed})
    write_json(identity.config.output_dir / "training_progress.json", identity.progress)


def complete_training(identity: RunIdentity, model: Any, result: Any, tracking: Any) -> dict[str, Any]:
    """Verify the produced checkpoints, then publish the immutable run summary."""
    observed = observe_training(identity, model, result)
    summary = build_summary_payload(identity, observed)
    publish_summary(identity, observed, summary, tracking)
    return summary
