"""Explicit official Carparts-Seg preparation with a reproducibility manifest."""

from pathlib import Path
from typing import Any

from vision.carparts_dataset.constants import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST,
    MANIFEST_NAME,
    OFFICIAL_ARCHIVE,
)
from vision.carparts_dataset.inspection import class_mapping, inspect_carparts_dataset, read_yaml, require_valid
from vision.carparts_dataset.manifest import build_manifest
from vision.dataset import SPLITS, DatasetValidationError
from vision.training import write_json


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
    from ultralytics.utils import ROOT as ULTRALYTICS_ROOT

    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    official_yaml = ULTRALYTICS_ROOT / "cfg/datasets/carparts-seg.yaml"
    official_document = read_yaml(official_yaml)
    if official_document.get("download") != OFFICIAL_ARCHIVE:
        raise DatasetValidationError(
            "Installed official Carparts-Seg YAML changed its download URL; review before downloading"
        )
    _ensure_dataset(dataset_root)
    source_yaml = _source_yaml(dataset_root, official_yaml)
    mapping = class_mapping(read_yaml(source_yaml))
    if mapping != class_mapping(official_document):
        raise DatasetValidationError("Archive classes differ from the installed official dataset YAML")
    copied_source = dataset_root / "source_dataset.yaml"
    copied_source.write_bytes(source_yaml.read_bytes())
    dataset_yaml = _write_training_yaml(dataset_root, mapping)
    inspection = inspect_carparts_dataset(dataset_yaml)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path.with_name("carparts_dataset_inspection.json"), inspection)
    require_valid(inspection)
    manifest = build_manifest(inspection, _paths(dataset_root, copied_source, official_yaml, dataset_yaml))
    write_json(dataset_root / MANIFEST_NAME, manifest)
    write_json(manifest_path, manifest)
    return manifest


def _ensure_dataset(dataset_root):
    if dataset_root.is_dir():
        return
    from ultralytics.utils.downloads import safe_download

    extracted = Path(safe_download(OFFICIAL_ARCHIVE, dir=dataset_root.parent, unzip=True, delete=False))
    if extracted.resolve() != dataset_root:
        raise DatasetValidationError(f"Official archive extracted to an unexpected root: {extracted}")


def _source_yaml(dataset_root, official_yaml):
    candidates = [dataset_root / name for name in ("data.yaml", "carparts-seg.yaml") if (dataset_root / name).is_file()]
    return candidates[0] if candidates else official_yaml


def _write_training_yaml(dataset_root, mapping):
    dataset_yaml = dataset_root / "dataset.yaml"
    normalized = {
        "path": str(dataset_root),
        **{split: f"images/{split}" for split in SPLITS},
        "names": {item["yolo_class_id"]: item["name"] for item in mapping},
        "nc": len(mapping),
    }
    write_json(dataset_yaml, normalized)
    return dataset_yaml


def _paths(dataset_root, copied_source, official_yaml, dataset_yaml):
    return {
        "dataset_root": dataset_root,
        "copied_source": copied_source,
        "official_yaml": official_yaml,
        "dataset_yaml": dataset_yaml,
    }
