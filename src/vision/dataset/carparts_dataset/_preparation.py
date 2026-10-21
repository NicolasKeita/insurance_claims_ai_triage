"""Preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.damage.training import file_sha256, write_json
from vision.dataset import SPLITS, DatasetValidationError
from vision.dataset.carparts_dataset._constants import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST,
    MANIFEST_NAME,
    OFFICIAL_ARCHIVE,
    OFFICIAL_SOURCE,
)
from vision.dataset.carparts_dataset._inspection import inspect_carparts_dataset
from vision.dataset.carparts_dataset._schema import _class_mapping, _read_yaml, _require_valid


def prepare_carparts_dataset(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """
    Explicit official download followed by real inspection and reproducibility manifest.

    Existing folders are inspected below, never considered proof of success.

    JSON is valid YAML; no download directive.
    """
    dataset_root, official_yaml, official_document, ultralytics_version = _official_carparts(dataset_root)
    copied_source, dataset_yaml = _normalize_carparts_yaml(dataset_root, official_yaml, official_document)
    manifest_path = Path(manifest_path).resolve()
    inspection = inspect_carparts_dataset(dataset_yaml)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path.with_name("carparts_dataset_inspection.json"), inspection)
    _require_valid(inspection)
    manifest = _carparts_manifest(inspection, dataset_root, copied_source, ultralytics_version, official_yaml)
    write_json(dataset_root / MANIFEST_NAME, manifest)
    write_json(manifest_path, manifest)
    return manifest


def _official_carparts(dataset_root):
    from ultralytics.utils import ROOT as ULTRALYTICS_ROOT
    from ultralytics.utils.downloads import safe_download

    import ultralytics

    dataset_root = Path(dataset_root).resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    official_yaml = ULTRALYTICS_ROOT / "cfg/datasets/carparts-seg.yaml"
    official_document = _read_yaml(official_yaml)
    if official_document.get("download") != OFFICIAL_ARCHIVE:
        raise DatasetValidationError(
            "Installed official Carparts-Seg YAML changed its download URL; review before downloading"
        )
    if not dataset_root.is_dir():
        extracted = Path(safe_download(OFFICIAL_ARCHIVE, dir=dataset_root.parent, unzip=True, delete=False))
        if extracted.resolve() != dataset_root:
            raise DatasetValidationError(f"Official archive extracted to an unexpected root: {extracted}")
    return dataset_root, official_yaml, official_document, ultralytics.__version__


def _normalize_carparts_yaml(dataset_root, official_yaml, official_document):
    candidates = [dataset_root / name for name in ("data.yaml", "carparts-seg.yaml") if (dataset_root / name).is_file()]
    source_yaml = candidates[0] if candidates else official_yaml
    source_document = _read_yaml(source_yaml)
    mapping = _class_mapping(source_document)
    if mapping != _class_mapping(official_document):
        raise DatasetValidationError("Archive classes differ from the installed official dataset YAML")
    copied_source = dataset_root / "source_dataset.yaml"
    copied_source.write_bytes(source_yaml.read_bytes())
    dataset_yaml = dataset_root / "dataset.yaml"
    normalized = {
        "path": str(dataset_root),
        **{split: f"images/{split}" for split in SPLITS},
        "names": {item["yolo_class_id"]: item["name"] for item in mapping},
        "nc": len(mapping),
    }
    write_json(dataset_yaml, normalized)
    return copied_source, dataset_yaml


def _carparts_manifest(inspection, dataset_root, copied_source, ultralytics_version, official_yaml):
    license_files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and ("license" in path.name.lower() or "readme" in path.name.lower())
    )
    return inspection | {
        "schema_version": 1,
        "source": OFFICIAL_SOURCE,
        "source_name": "Ultralytics official Carparts-Seg",
        "source_repository": OFFICIAL_ARCHIVE,
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "evidence": "CC BY 4.0 badge on Ultralytics dataset documentation",
            "attribution": "Gianmarco Russo; car-seg dataset on Roboflow Universe (2023)",
            "archive_metadata_files": [path.relative_to(dataset_root).as_posix() for path in license_files],
        },
        "ultralytics_version": str(ultralytics_version),
        "source_dataset_yaml_sha256": file_sha256(copied_source),
        "installed_official_yaml_sha256": file_sha256(official_yaml),
        "relevant_files": {
            "source_yaml": str(copied_source),
            "training_yaml": str(dataset_root / "dataset.yaml"),
            "installed_official_yaml": str(official_yaml),
            "training_config": "configs/vision/carparts_yolo26n_seg_v1.toml",
        },
        "metrics_scope": "Official Carparts-Seg supplied TEST; TRAIN/VAL used for optimization/checkpoint selection.",
    }
