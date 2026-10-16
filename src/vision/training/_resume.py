"""Resume validation: refuse a different, completed or already evaluated baseline."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vision.training._artifacts import dataset_provenance, file_sha256, verify_class_names
from vision.training._config import DamageTrainingConfig


def _require_uncompleted_run(config: DamageTrainingConfig) -> None:
    for name in ("training_summary.json", "test_metrics.json", "evaluation", "test_smoke"):
        if (config.output_dir / name).exists():
            raise ValueError("Resume refused: completed training/evaluation artifacts already exist")


def _validate_resume_epochs(config: DamageTrainingConfig, metadata: dict[str, Any]) -> None:
    completed, best_epoch = metadata.get("epochs_completed"), metadata.get("best_epoch")
    if (
        type(completed) is not int
        or not 0 < completed < config.epochs
        or type(best_epoch) is not int
        or not 0 < best_epoch <= completed
    ):
        raise ValueError("Resume epochs must identify an incomplete baseline and its selected best epoch")
    elapsed = metadata.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Resume metadata requires a finite elapsed training duration")


def _validate_resume_checkpoints(config: DamageTrainingConfig, metadata: dict[str, Any]) -> None:
    for name in ("best", "last"):
        checkpoint = config.output_dir / f"train/weights/{name}.pt"
        if not checkpoint.is_file() or file_sha256(checkpoint) != metadata.get(f"{name}_checkpoint_sha256"):
            raise ValueError(f"Resume {name}.pt SHA-256 differs from validated recovery metadata")
    base = Path(metadata.get("base_checkpoint", ""))
    if not base.is_file() or file_sha256(base) != metadata.get("base_checkpoint_sha256"):
        raise ValueError("Resume original pretrained checkpoint SHA-256 differs from recovery metadata")


def _validate_resume_metadata(
    config: DamageTrainingConfig,
    manifest: dict[str, Any],
    threshold: float,
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    if metadata.get("config") != config.payload():
        raise ValueError("Resume configuration differs from the interrupted baseline")
    if metadata.get("dataset") != provenance_provider(config.dataset_yaml, manifest):
        raise ValueError("Resume dataset provenance differs from the interrupted baseline")
    if metadata.get("confidence_threshold") != threshold:
        raise ValueError("Resume confidence threshold differs from the interrupted baseline")
    if metadata.get("checkpoint_selection_split", "val") != "val":
        raise ValueError("Resume requires a VAL-selected baseline")
    _validate_resume_epochs(config, metadata)
    _validate_resume_checkpoints(config, metadata)
def load_resume_metadata(
    config: DamageTrainingConfig,
    manifest: dict[str, Any],
    threshold: float,
    *,
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refuse to resume a different, completed, or already evaluated baseline."""
    _require_uncompleted_run(config)
    path = config.output_dir / "recovery_training_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Validated recovery metadata required before resume: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    _validate_resume_metadata(config, manifest, threshold, provenance_provider or dataset_provenance, metadata)
    return metadata


def validate_resume_checkpoint(
    model: Any,
    config: DamageTrainingConfig,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Inspect the loaded local checkpoint without permitting scratch fallback."""
    checkpoint = getattr(model, "ckpt", None)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("optimizer") is None
        or checkpoint.get("epoch") != metadata["epochs_completed"] - 1
    ):
        raise ValueError("last.pt must retain optimizer state and the validated incomplete epoch")
    args = checkpoint.get("train_args", {})
    expected = {key: getattr(config, key) for key in ("epochs", "imgsz", "batch", "patience", "seed", "workers")}
    expected.update({"task": "segment", "split": "val", "val": True, "pretrained": True})
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("last.pt training arguments differ from the configured TRAIN/VAL baseline")
    if config.optimizer != "auto" and args.get("optimizer") != config.optimizer:
        raise ValueError("last.pt optimizer differs from the configured baseline")
    for key, expected_path in (("data", config.dataset_yaml), ("project", config.output_dir)):
        if Path(args.get(key, "")).resolve() != expected_path.resolve():
            raise ValueError(f"last.pt {key} path differs from the interrupted baseline")
    if args.get("name") != "train":
        raise ValueError("last.pt must belong to the exact train run directory")
    verify_class_names(getattr(model, "names", None), manifest["category_mapping"])
    return checkpoint


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