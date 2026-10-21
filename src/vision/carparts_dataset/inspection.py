"""Read-only Carparts-Seg validation and split counting."""

from pathlib import Path
from typing import Any

from vision.carparts_dataset.constants import DEFAULT_DATASET_YAML, PREPARATION_VERSION
from vision.carparts_dataset.scanning import _inspect_split
from vision.carparts_dataset.yaml_docs import class_mapping, read_yaml
from vision.dataset import SPLITS, DatasetValidationError
from vision.training import file_sha256


def inspect_carparts_dataset(dataset_yaml: Path = DEFAULT_DATASET_YAML) -> dict[str, Any]:
    """Validate all actual images/YOLO polygons and count the supplied splits.

    This never invokes an Ultralytics downloader, even for a missing directory.
    Label hashes are aggregated per split; images are decoded but not hashed.
    """
    dataset_yaml = Path(dataset_yaml).resolve()
    document = read_yaml(dataset_yaml)
    mapping = class_mapping(document)
    root = _dataset_root(document, dataset_yaml)
    report = _new_report(dataset_yaml, root, mapping)
    seen_images: set[Path] = set()
    for split in SPLITS:
        _inspect_split(report, split, (document, root, mapping, seen_images))
    _finalize_report(report)
    return report


def _dataset_root(document, dataset_yaml) -> Path:
    root = Path(document.get("path") or dataset_yaml.parent)
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    return root.resolve()


def _new_report(dataset_yaml, root, mapping) -> dict[str, Any]:
    return {
        "dataset_name": "Carparts-Seg",
        "dataset_root": str(root),
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": file_sha256(dataset_yaml),
        "category_mapping": mapping,
        "splits": {},
        "prepared_files": {},
        "annotation_hashes": {},
        "invalid_labels": [],
        "invalid_images": [],
        "missing_labels": [],
        "orphan_labels": [],
        "split_path_errors": [],
        "official_splits_used": True,
        "preparation_version": PREPARATION_VERSION,
    }


def _finalize_report(report):
    report["actual_sample_count"] = sum(value["image_count"] for value in report["splits"].values())
    source_names = {
        split: {Path(entry["image"]).name.split(".rf.")[0] for entry in report["prepared_files"][split]}
        for split in SPLITS
    }
    report["source_filename_overlap_audit"] = {
        "method": "Filename prefix before .rf.; repeated source identifiers only, no image hashes or pixel comparison.",
        "unique_source_identifiers": {split: len(values) for split, values in source_names.items()},
        "pairwise_shared_identifiers": {
            f"{left}_{right}": len(source_names[left] & source_names[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "limitation": "Supplied splits are preserved. Shared filename identifiers suggest related/augmented "
        "source images across splits; they do not prove identical pixels or independent unseen-photo "
        "generalization.",
    }
    report["valid"] = all(value["image_count"] > 0 for value in report["splits"].values()) and not any(
        report[key]
        for key in ("invalid_labels", "invalid_images", "missing_labels", "orphan_labels", "split_path_errors")
    )


def require_valid(report: dict[str, Any]) -> None:
    if not report["valid"]:
        counts = {
            key: len(report[key])
            for key in ("invalid_labels", "invalid_images", "missing_labels", "orphan_labels", "split_path_errors")
        }
        raise DatasetValidationError(
            f"Carparts-Seg inspection failed: {counts}; inspect the saved report before training"
        )
