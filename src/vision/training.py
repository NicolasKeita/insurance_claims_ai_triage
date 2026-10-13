"""Explicit training orchestration; importing this module loads no CV framework."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import tomllib
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    count = int(torch.cuda.device_count()) if available else 0
    return {
        "python_version": platform.python_version(),
        "ultralytics_version": str(ultralytics.__version__),
        "torch_version": str(torch.__version__),
        "cuda_available": available,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(count)],
    }


def disable_builtin_mlflow_callbacks(trainer: Any) -> None:
    """The built-in callback logs image batches. Keep only our metadata logger."""
    for event, callbacks in trainer.callbacks.items():
        trainer.callbacks[event] = [
            callback for callback in callbacks
            if not getattr(callback, "__module__", "").startswith("ultralytics.utils.callbacks.mlflow")
        ]


def run_training(
    config: DamageTrainingConfig,
    *,
    model_factory: Callable[[str], Any],
    environment_provider: Callable[[], dict[str, Any]],
    dataset_loader: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train on TRAIN; Ultralytics chooses best/early stopping from VAL only.

    The factory is intentionally injected. The CLI imports Ultralytics only after
    the prepared dataset has passed validation; tests use a lightweight stub.
    """
    if dataset_loader is None:
        from vision.dataset import load_prepared_dataset
        dataset_loader = load_prepared_dataset
    manifest = dataset_loader(config.dataset_yaml)
    if config.output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {config.output_dir}. Choose a new version/output_dir.")
    environment = environment_provider()
    model = model_factory(config.base_model)
    if getattr(model, "task", None) != "segment":
        raise ValueError("Base weights must contain an instance segmentation model (task=segment)")
    model.add_callback("on_pretrain_routine_start", disable_builtin_mlflow_callbacks)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    from vision.config import VisionSettings
    from vision.evaluation import extract_segmentation_metrics
    from vision.tracking import vision_tracking_run
    threshold = VisionSettings.from_env().confidence_threshold
    started = time.perf_counter()
    with vision_tracking_run(config) as tracking:
        result = model.train(
            data=str(config.dataset_yaml), epochs=config.epochs, imgsz=config.imgsz,
            batch=config.batch, patience=config.patience, seed=config.seed,
            device=config.device, workers=config.workers, optimizer=config.optimizer,
            project=str(config.output_dir), name="train", exist_ok=False,
            pretrained=True, val=True, save=True, plots=False,
        )
        trainer = model.trainer
        expected_best = config.output_dir / "train/weights/best.pt"
        actual_best = Path(trainer.best).resolve()
        if actual_best != expected_best.resolve() or not actual_best.is_file():
            raise RuntimeError(f"Training did not produce the expected selected checkpoint: {expected_best}")
        verify_class_names(getattr(model, "names", None), manifest["category_mapping"])
        actual_device = getattr(trainer, "device", None)
        environment["device_used"] = str(actual_device) if actual_device is not None else None
        epoch = getattr(trainer, "epoch", None)
        epochs_completed = int(epoch) + 1 if epoch is not None else None
        metrics = extract_segmentation_metrics(result, manifest["category_mapping"], manifest["splits"]["val"])
        summary = {
            "schema_version": 1, "status": "completed", "task": "segment",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config.payload(), "environment": environment,
            "dataset": dataset_provenance(config.dataset_yaml, manifest),
            "confidence_threshold": threshold,
            "checkpoint_selection_split": "val",
            "best_checkpoint": "train/weights/best.pt",
            "best_checkpoint_sha256": file_sha256(actual_best),
            "epochs_completed": epochs_completed,
            "stopped_before_requested_epochs": epochs_completed < config.epochs if epochs_completed is not None else None,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_metrics": metrics,
            "reproducibility": "seeded / reproducible configuration; bitwise GPU determinism is not guaranteed",
            "registry": "NOT EXECUTED — custom vision registry packaging deferred; no production alias",
        }
        summary_path = config.output_dir / "training_summary.json"
        write_json(summary_path, summary)
        if tracking is not None:
            tracking.log_training(summary, summary_path, actual_best)
            summary["mlflow_run_id"] = tracking.run_id
            write_json(summary_path, summary)
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
        provenance.update({key: manifest.get(key) for key in (
            "source_name", "source_repository", "source_revision", "actual_sample_count",
            "annotation_fields", "official_splits_used", "split_version",
            "split_manifest_sha256", "metrics_scope",
        )})
    return provenance


def verify_class_names(names: Any, mapping: list[dict[str, Any]]) -> None:
    expected = {int(item["yolo_class_id"]): item["name"] for item in mapping}
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or {int(key): value for key, value in names.items()} != expected:
        raise ValueError("Checkpoint class names/order do not match the prepared CarDD taxonomy")
