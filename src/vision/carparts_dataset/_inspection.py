"""Inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.carparts_dataset._constants import DEFAULT_DATASET_YAML, PREPARATION_VERSION
from vision.carparts_dataset._inventory import _inspect_carparts_splits
from vision.carparts_dataset._schema import _class_mapping, _read_yaml
from vision.dataset import SPLITS
from vision.training import file_sha256


def inspect_carparts_dataset(dataset_yaml: Path = DEFAULT_DATASET_YAML) -> dict[str, Any]:
    """Validate all actual images/YOLO polygons and count the supplied splits.

    This never invokes an Ultralytics downloader, even for a missing directory.
    Label hashes are aggregated per split; images are decoded but not hashed.
    """

    dataset_yaml = Path(dataset_yaml).resolve()
    document = _read_yaml(dataset_yaml)
    mapping = _class_mapping(document)
    root = Path(document.get("path") or dataset_yaml.parent)
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    root = root.resolve()
    report = _inspection_metadata(dataset_yaml, root, mapping, document)
    _inspect_carparts_splits(document, root, mapping, report)
    report["actual_sample_count"] = sum(value["image_count"] for value in report["splits"].values())
    _source_overlap_audit(report)
    report["valid"] = all(value["image_count"] > 0 for value in report["splits"].values()) and not any(
        report[key]
        for key in ("invalid_labels", "invalid_images", "missing_labels", "orphan_labels", "split_path_errors")
    )
    return report


def _inspection_metadata(dataset_yaml, root, mapping, document):
    report: dict[str, Any] = {
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
    return report


def _source_overlap_audit(report):
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
