"""Preparation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from vision.dataset import (
    PREPARED_MANIFEST_NAME,
    SPLITS,
    DatasetValidationError,
    _try_ultralytics_conversion,
    _write_json,
    annotation_hash,
    load_prepared_dataset,
)
from vision.dataset.cardd_hf_preparation._constants import DEFAULT_HF_MANIFEST_PATH
from vision.dataset.cardd_hf_preparation._fingerprint import hf_prepared_fingerprint
from vision.dataset.cardd_hf_preparation._manifest import _hf_manifest
from vision.dataset.cardd_hf_preparation.conversion._conversion import _convert_hf_splits
from vision.dataset.cardd_hf_preparation.conversion._files import _hf_output


def prepare_cardd_hf_yolo(
    dataset: Any,
    output_dir: Path | str | None = None,
    *,
    split_manifest_path: Path | str | None = None,
    manifest_path: Path | str | None = DEFAULT_HF_MANIFEST_PATH,
    converter: str = "auto",
) -> dict[str, Any]:
    """Prepare every inspected instance using a persisted, immutable custom split.

    Annotation failures abort; this mirror workflow never silently drops damage
    instances. Empty images, when present in the inspected source, retain empty
    labels. Reusing a split version validates the source and does not regenerate it.
    """
    from vision.dataset.cardd_hf_split import (
        DEFAULT_SPLIT_MANIFEST_PATH,
        create_or_load_cardd_hf_split,
    )

    output = _hf_output(dataset, output_dir, manifest_path, converter)
    split_document = create_or_load_cardd_hf_split(
        dataset, manifest_path=split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH
    )
    classes, category_mapping, manifest = _hf_manifest(dataset, split_document)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cardd-hf-preparation-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        documents, expected = _convert_hf_splits(dataset, split_document, manifest, stage)
        conversion = _hf_backend(stage, documents, expected, classes, converter)
        manifest["conversion"] = conversion
        dataset_yaml = _validate_hf_stage(stage, manifest, category_mapping, classes)
        dataset_yaml["path"] = str(output)
        _write_json(stage / "dataset.yaml", dataset_yaml)
        stage.rename(output)
    if manifest_path is not None:
        _write_json(Path(manifest_path), manifest)
    return manifest


def _hf_backend(stage, documents, expected, classes, converter):
    conversion = {
        "backend": "native_polygon",
        "unsupported_policy": "error",
        "segmentation_enabled": True,
        "bbox_fallback": False,
        "source_format": "FiftyOne normalized instance polygons",
        "multipart_policy": "preserve component vertices using retraced nearest-vertex bridges",
        "multipart_limitation": "YOLO uses one polygon per instance; "
        "disconnected topology and holes cannot be represented exactly, "
        "and rasterization may include bridge pixels",
        "use_segments": True,
        "cls91to80": False,
    }
    if converter != "native":
        try:
            _try_ultralytics_conversion(stage, documents, expected, len(classes))
            conversion["backend"] = "ultralytics.convert_coco"
        except Exception as exc:
            if converter == "ultralytics":
                raise DatasetValidationError(f"COCO-format converter failed HF polygon validation: {exc}") from exc
            conversion["fallback_reason"] = (
                f"{type(exc).__name__}: converter unavailable, incompatible, or failed strict polygon validation"
            )
    return conversion


def _validate_hf_stage(stage, manifest, category_mapping, classes):
    for entries in manifest["prepared_files"].values():
        for entry in entries:
            entry["label_sha256"] = annotation_hash(stage / entry["label"])
    manifest["dataset_fingerprint"] = hf_prepared_fingerprint(manifest, category_mapping)
    dataset_yaml = {
        "path": str(stage),
        **{name: f"images/{name}" for name in SPLITS},
        "names": classes,
        "nc": len(classes),
    }
    _write_json(stage / "dataset.yaml", dataset_yaml)
    _write_json(stage / PREPARED_MANIFEST_NAME, manifest)
    load_prepared_dataset(stage / "dataset.yaml")
    for scratch in (stage / "converter_annotations", stage / "converter_output"):
        if scratch.exists():
            shutil.rmtree(scratch)
    return dataset_yaml
