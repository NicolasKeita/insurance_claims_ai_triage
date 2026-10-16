"""Checksums, environment snapshots, JSON artifacts and portable provenance."""

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


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


def observe_environment(torch: Any, ultralytics: Any) -> dict[str, Any]:
    """Record the exact interpreter, framework builds and visible CUDA devices."""
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
        raise ValueError("Checkpoint class names/order do not match the prepared dataset taxonomy")