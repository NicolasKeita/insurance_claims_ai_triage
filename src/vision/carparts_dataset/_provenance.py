"""Provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision.carparts_dataset._constants import (
    DEFAULT_DATASET_YAML,
    MANIFEST_NAME,
    PREPARATION_VERSION,
)
from vision.carparts_dataset._inspection import inspect_carparts_dataset
from vision.carparts_dataset._schema import _require_valid
from vision.dataset import DatasetValidationError
from vision.training import file_sha256


def load_carparts_dataset(dataset_yaml: Path = DEFAULT_DATASET_YAML) -> dict[str, Any]:
    """Reinspect local data and verify the frozen manifest without downloading."""
    dataset_yaml = Path(dataset_yaml).resolve()
    path = dataset_yaml.parent / MANIFEST_NAME
    if not path.is_file():
        raise DatasetValidationError(
            f"Prepared Carparts-Seg manifest missing: {path}; run the explicit preparation script"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inspection = inspect_carparts_dataset(dataset_yaml)
    _require_valid(inspection)
    if manifest.get("schema_version") != 1 or manifest.get("preparation_version") != PREPARATION_VERSION:
        raise DatasetValidationError("Unsupported Carparts-Seg preparation manifest version")
    for key in (
        "dataset_yaml_sha256",
        "category_mapping",
        "splits",
        "annotation_hashes",
        "prepared_files",
        "actual_sample_count",
    ):
        if manifest.get(key) != inspection[key]:
            raise DatasetValidationError(f"Carparts-Seg {key} changed since preparation")
    return manifest


def carparts_provenance(dataset_yaml: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source",
        "source_name",
        "source_repository",
        "dataset_name",
        "license",
        "official_splits_used",
        "preparation_version",
        "category_mapping",
        "splits",
        "annotation_hashes",
        "actual_sample_count",
        "source_dataset_yaml_sha256",
        "installed_official_yaml_sha256",
        "ultralytics_version",
        "metrics_scope",
        "source_filename_overlap_audit",
    )
    return {key: manifest[key] for key in keys if key in manifest} | {
        "dataset_yaml_sha256": file_sha256(dataset_yaml),
        "prepared_manifest_sha256": file_sha256(Path(dataset_yaml).parent / MANIFEST_NAME),
    }
