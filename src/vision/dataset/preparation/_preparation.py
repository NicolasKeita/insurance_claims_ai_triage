"""Preparation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from vision.dataset._constants import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PREPARED_DIR,
    PREPARED_MANIFEST_NAME,
    SPLITS,
)
from vision.dataset._files import _write_json, annotation_hash
from vision.dataset._models import CarddDataset, DatasetValidationError
from vision.dataset.coco._inspection import build_dataset_manifest
from vision.dataset.preparation._conversion import _convert_coco_splits
from vision.dataset.preparation._converter import _coco_backend
from vision.dataset.preparation._output import _preparation_output
from vision.dataset.prepared._prepared import load_prepared_dataset


def prepare_cardd_yolo(
    dataset: CarddDataset,
    output_dir: Path = DEFAULT_PREPARED_DIR,
    *,
    manifest_path: Path | None = DEFAULT_MANIFEST_PATH,
    unsupported: str = "error",
    converter: str = "auto",
) -> dict[str, Any]:
    """Prepare atomically, with explicit unsupported mask policy and validation.

    ``error`` rejects crowd, RLE, missing, malformed and duplicate segmentations.
    ``skip`` records every excluded reason and count. Neither creates fake masks.
    Multipart polygons preserve component vertices with retraced bridges because
    YOLO supports only one polygon per instance; this limitation is recorded.
    ``auto`` uses the official converter if available and output-compatible, then
    records a native polygon-only fallback when it is absent or incompatible.

    Two different source names must never accidentally collapse into one class.

    JSON is valid YAML and avoids a YAML dependency for preparation.

    Converter scratch data is not part of the prepared dataset.

    Ultralytics resolves relative dataset paths against its datasets_dir, not the YAML directory. Only portable
    manifest metadata excludes root.
    """
    output = _preparation_output(dataset, output_dir, manifest_path, unsupported, converter)
    manifest, class_ids = _preparation_manifest(dataset)
    manifest["prepared_files"] = {}
    _stage_coco_dataset(dataset, manifest, output, unsupported, converter)
    if manifest_path is not None:
        _write_json(Path(manifest_path), manifest)
    return manifest


def _preparation_manifest(dataset):
    from vision.taxonomy import map_category_name

    manifest = build_dataset_manifest(dataset)
    for category in manifest["category_mapping"]:
        try:
            category["damage_type"] = map_category_name(category["name"]).value
        except ValueError as exc:
            raise DatasetValidationError(f"Unsupported domain taxonomy category: {category['name']}") from exc
    if len({item["damage_type"] for item in manifest["category_mapping"]}) != len(manifest["category_mapping"]):
        raise DatasetValidationError("COCO categories collapse to duplicate domain damage types")
    class_ids = {item["category_id"]: item["yolo_class_id"] for item in manifest["category_mapping"]}
    return manifest, class_ids


def _validate_coco_stage(stage, manifest, class_ids):
    for files in manifest["prepared_files"].values():
        for entry in files:
            entry["label_sha256"] = annotation_hash(stage / entry["label"])
    dataset_yaml = {
        "path": str(stage),
        **{name: f"images/{name}" for name in SPLITS},
        "names": [item["name"] for item in manifest["category_mapping"]],
        "nc": len(class_ids),
    }
    _write_json(stage / "dataset.yaml", dataset_yaml)
    _write_json(stage / PREPARED_MANIFEST_NAME, manifest)
    load_prepared_dataset(stage / "dataset.yaml")
    for scratch in (stage / "converter_annotations", stage / "converter_output"):
        if scratch.exists():
            shutil.rmtree(scratch)
    return dataset_yaml


def _stage_coco_dataset(dataset, manifest, output, unsupported, converter):
    class_ids = {item["category_id"]: item["yolo_class_id"] for item in manifest["category_mapping"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cardd-preparation-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "dataset"
        stage.mkdir()
        expected, documents = _convert_coco_splits(dataset, manifest, stage, class_ids, unsupported)
        conversion = _coco_backend(stage, documents, expected, converter, unsupported)
        manifest["conversion"] = conversion
        dataset_yaml = _validate_coco_stage(stage, manifest, class_ids)
        dataset_yaml["path"] = str(output)
        _write_json(stage / "dataset.yaml", dataset_yaml)
        stage.rename(output)
