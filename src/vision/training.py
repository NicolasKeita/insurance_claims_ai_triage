"""Explicit training orchestration; importing this module loads no CV framework."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def observe_environment(torch: Any, ultralytics: Any) -> dict[str, Any]:
    """Record observed libraries/hardware; modules are supplied by explicit scripts."""
    available = bool(torch.cuda.is_available())
    count_reader = getattr(torch.cuda, "device_count", None)
    count = int(count_reader()) if callable(count_reader) else 0
    usable_count = count if available else 0
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "ultralytics_version": str(ultralytics.__version__),
        "torch_version": str(torch.__version__),
        "cuda_available": available,
        "cuda_device_count": count,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(usable_count)],
        "cuda_device_properties": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                "total_memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 3),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(usable_count)
        ],
    }


def disable_builtin_mlflow_callbacks(trainer: Any) -> None:
    """The built-in callback logs image batches. Keep only our metadata logger."""
    for event, callbacks in trainer.callbacks.items():
        trainer.callbacks[event] = [
            callback
            for callback in callbacks
            if not getattr(callback, "__module__", "").startswith("ultralytics.utils.callbacks.mlflow")
        ]


def load_resume_metadata(
    config: DamageTrainingConfig,
    manifest: dict[str, Any],
    threshold: float,
    *,
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refuse to resume a different, completed, or already evaluated baseline."""
    for name in ("training_summary.json", "test_metrics.json", "evaluation", "test_smoke"):
        if (config.output_dir / name).exists():
            raise ValueError("Resume refused: completed training/evaluation artifacts already exist")
    path = config.output_dir / "recovery_training_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Validated recovery metadata required before resume: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("config") != config.payload():
        raise ValueError("Resume configuration differs from the interrupted baseline")
    provenance_provider = provenance_provider or dataset_provenance
    if metadata.get("dataset") != provenance_provider(config.dataset_yaml, manifest):
        raise ValueError("Resume dataset provenance differs from the interrupted baseline")
    if metadata.get("confidence_threshold") != threshold:
        raise ValueError("Resume confidence threshold differs from the interrupted baseline")
    if metadata.get("checkpoint_selection_split", "val") != "val":
        raise ValueError("Resume requires a VAL-selected baseline")
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
    for name in ("best", "last"):
        checkpoint = config.output_dir / f"train/weights/{name}.pt"
        if not checkpoint.is_file() or file_sha256(checkpoint) != metadata.get(f"{name}_checkpoint_sha256"):
            raise ValueError(f"Resume {name}.pt SHA-256 differs from validated recovery metadata")
    base = Path(metadata.get("base_checkpoint", ""))
    if not base.is_file() or file_sha256(base) != metadata.get("base_checkpoint_sha256"):
        raise ValueError("Resume original pretrained checkpoint SHA-256 differs from recovery metadata")
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
    expected = {
        key: getattr(config, key)
        for key in (
            "epochs",
            "imgsz",
            "batch",
            "patience",
            "seed",
            "workers",
        )
    }
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


def run_training(
    config: DamageTrainingConfig,
    *,
    model_factory: Callable[[str], Any],
    environment_provider: Callable[[], dict[str, Any]],
    dataset_loader: Callable[[Path], dict[str, Any]] | None = None,
    resume: bool = False,
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
    confidence_threshold: float | None = None,
    tracking_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Train on TRAIN; Ultralytics chooses best/early stopping from VAL only.

    The factory is intentionally injected. The CLI imports Ultralytics only after
    the prepared dataset has passed validation; tests use a lightweight stub.
    """
    if dataset_loader is None:
        from vision.dataset import load_prepared_dataset

        dataset_loader = load_prepared_dataset
    manifest = dataset_loader(config.dataset_yaml)
    from vision.config import VisionSettings

    threshold = VisionSettings.from_env().confidence_threshold if confidence_threshold is None else confidence_threshold
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be finite and between zero and one")
    provenance_provider = provenance_provider or dataset_provenance
    recovery = (
        load_resume_metadata(config, manifest, threshold, provenance_provider=provenance_provider) if resume else None
    )
    if config.output_dir.exists() and not resume:
        raise FileExistsError(f"Run directory already exists: {config.output_dir}. Choose a new version/output_dir.")
    environment = environment_provider()
    initial_checkpoint = str(config.output_dir / "train/weights/last.pt") if resume else config.base_model
    model = model_factory(initial_checkpoint)
    if getattr(model, "task", None) != "segment":
        raise ValueError("Base weights must contain an instance segmentation model (task=segment)")
    model.add_callback("on_pretrain_routine_start", disable_builtin_mlflow_callbacks)
    saved_checkpoint = validate_resume_checkpoint(model, config, recovery, manifest) if recovery else None
    if not resume:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {"best_epoch": recovery["best_epoch"] if recovery else None}
    provenance = provenance_provider(config.dataset_yaml, manifest)
    previous_elapsed = float(recovery["elapsed_seconds"]) if recovery else 0.0
    base_checkpoint = Path(
        recovery["base_checkpoint"] if recovery else getattr(model, "ckpt_path", config.base_model)
    ).resolve()
    base_checkpoint_sha256 = (
        recovery["base_checkpoint_sha256"]
        if recovery
        else (file_sha256(base_checkpoint) if base_checkpoint.is_file() else None)
    )
    started = time.perf_counter()

    def restore_resume_stopper(trainer: Any) -> None:
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

    if recovery:
        model.add_callback("on_pretrain_routine_end", restore_resume_stopper)

    def record_checkpoint_progress(trainer: Any) -> None:
        # save_model writes best.pt whenever fitness equals best_fitness,
        # including ties. The stopper records only strict improvements, so its
        # best_epoch can differ from the epoch of the actual selected weights.
        epoch_completed = int(trainer.epoch) + 1
        if trainer.fitness == trainer.best_fitness:
            progress["best_epoch"] = epoch_completed
        progress.update(
            {
                "status": "running",
                "epochs_completed": epoch_completed,
                "checkpoint_selection_split": "val",
                "fitness": float(trainer.fitness),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(config.output_dir / "training_progress.json", progress)
        recovery_state = {
            "config": config.payload(),
            "dataset": provenance,
            "confidence_threshold": threshold,
            "checkpoint_selection_split": "val",
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "best_checkpoint_sha256": file_sha256(Path(trainer.best)),
            "last_checkpoint_sha256": file_sha256(Path(trainer.last)),
            "best_epoch": progress["best_epoch"],
            "epochs_completed": epoch_completed,
            "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
            "resume_count": (recovery.get("resume_count", 0) if recovery else 0) + int(resume),
        }
        write_json(config.output_dir / "recovery_training_metadata.json", recovery_state)

    model.add_callback("on_model_save", record_checkpoint_progress)
    from vision.evaluation import extract_segmentation_metrics
    from vision.tracking import vision_tracking_run

    tracking_factory = tracking_factory or vision_tracking_run
    with tracking_factory(config) as tracking:
        training_arguments = dict(
            data=str(config.dataset_yaml),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            patience=config.patience,
            seed=config.seed,
            device=config.device,
            workers=config.workers,
            optimizer=config.optimizer,
            project=str(config.output_dir),
            name="train",
            exist_ok=False,
            pretrained=True,
            val=True,
            split="val",
            save=True,
            plots=True,
        )
        if recovery:
            training_arguments.update({"resume": initial_checkpoint, "save_dir": str(config.output_dir / "train")})
        result = model.train(**training_arguments)
        trainer = model.trainer
        expected_best = config.output_dir / "train/weights/best.pt"
        actual_best = Path(trainer.best).resolve()
        if actual_best != expected_best.resolve() or not actual_best.is_file():
            raise RuntimeError(f"Training did not produce the expected selected checkpoint: {expected_best}")
        expected_last = config.output_dir / "train/weights/last.pt"
        actual_last = Path(trainer.last).resolve()
        if actual_last != expected_last.resolve() or not actual_last.is_file():
            raise RuntimeError(f"Training did not produce the expected final checkpoint: {expected_last}")
        verify_class_names(getattr(model, "names", None), manifest["category_mapping"])
        actual_device = getattr(trainer, "device", None)
        environment["device_used"] = str(actual_device) if actual_device is not None else None
        epoch = getattr(trainer, "epoch", None)
        epochs_completed = int(epoch) + 1 if epoch is not None else None
        metrics = extract_segmentation_metrics(result, manifest["category_mapping"], manifest["splits"]["val"])
        stopper = getattr(trainer, "stopper", None)
        best_epoch = progress["best_epoch"]
        if best_epoch is None:
            raise RuntimeError("Training did not record the epoch that wrote the selected best.pt checkpoint")
        early_stopped = epochs_completed < config.epochs if epochs_completed is not None else None
        summary = {
            "schema_version": 1,
            "status": "completed",
            "task": "segment",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config.payload(),
            "environment": environment,
            "dataset": provenance,
            "confidence_threshold": threshold,
            "checkpoint_selection_split": "val",
            "training_optimization_split": "train",
            "training_validation_split": "val",
            "test_used_during_training": False,
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "best_checkpoint": "train/weights/best.pt",
            "best_checkpoint_sha256": file_sha256(actual_best),
            "last_checkpoint": "train/weights/last.pt",
            "last_checkpoint_sha256": file_sha256(actual_last),
            "best_epoch": best_epoch,
            "best_epoch_source": "on_model_save: actual best.pt write, including fitness ties",
            "early_stopping_best_epoch": getattr(stopper, "best_epoch", None),
            "epochs_completed": epochs_completed,
            "effective_batch_size": getattr(trainer, "batch_size", config.batch),
            "best_validation_fitness": float(trainer.best_fitness)
            if getattr(trainer, "best_fitness", None) is not None
            else None,
            "stopped_before_requested_epochs": early_stopped,
            "early_stopping_triggered": early_stopped,
            "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
            "elapsed_seconds_before_resume": previous_elapsed,
            "elapsed_seconds_this_execution": time.perf_counter() - started,
            "elapsed_seconds_method": "Sum of checkpoint-recorded prior duration and this execution; excludes interruption downtime. Native results.csv time restarts for resumed segments.",
            "resumed": resume,
            "resume_from_epoch": recovery["epochs_completed"] if recovery else None,
            "resume_input_checkpoint_sha256": recovery["last_checkpoint_sha256"] if recovery else None,
            "validation_metrics": metrics,
            "reproducibility": "seeded / reproducible configuration; bitwise GPU determinism is not guaranteed",
            "registry": "NOT EXECUTED — custom vision registry packaging deferred; no production alias",
            "training_artifacts": {
                "results_csv": "train/results.csv",
                "results_png": "train/results.png",
                "args_yaml": "train/args.yaml",
            },
        }
        summary_path = config.output_dir / "training_summary.json"
        write_json(summary_path, summary)
        if tracking is not None:
            tracking.log_training(summary, summary_path, actual_best)
            summary["mlflow_run_id"] = tracking.run_id
            write_json(summary_path, summary)
        progress.update({"status": "completed", "epochs_completed": epochs_completed})
        write_json(config.output_dir / "training_progress.json", progress)
    return summary


def dataset_provenance(dataset_yaml: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Portable provenance excludes the source dataset root."""
    provenance = {
        "dataset_yaml_sha256": file_sha256(dataset_yaml),
        "annotation_hashes": manifest["annotation_hashes"],
        "category_mapping": manifest["category_mapping"],
        "splits": manifest["splits"],
        "preparation_version": manifest.get("preparation_version"),
        "prepared_manifest_sha256": file_sha256(dataset_yaml.parent / "cardd_dataset_manifest.json"),
    }
    if manifest.get("source_name") == "CarDD HF mirror":
        provenance.update(
            {
                key: manifest.get(key)
                for key in (
                    "source_name",
                    "source_repository",
                    "source_revision",
                    "actual_sample_count",
                    "annotation_fields",
                    "official_splits_used",
                    "split_version",
                    "split_manifest_sha256",
                    "metrics_scope",
                )
            }
        )
    return provenance


def verify_class_names(names: Any, mapping: list[dict[str, Any]]) -> None:
    expected = {int(item["yolo_class_id"]): item["name"] for item in mapping}
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or {int(key): value for key, value in names.items()} != expected:
        raise ValueError("Checkpoint class names/order do not match the prepared dataset taxonomy")
