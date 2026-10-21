"""Manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from vision.cardd_hf_preparation._constants import HF_PREPARATION_VERSION
from vision.cardd_hf_preparation._fingerprint import _canonical_hash
from vision.dataset import (
    DatasetValidationError,
)


def _hf_manifest(dataset, split_document):
    classes, category_mapping, enriched_mapping = _hf_categories(dataset)
    manifest: dict[str, Any] = {
        "preparation_version": HF_PREPARATION_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_name": "CarDD HF mirror",
        "source_repository": dataset.repository,
        "source_revision": dataset.revision,
        "source_format": dataset.source_format,
        "actual_sample_count": len(dataset.samples),
        "actual_damage_classes": classes,
        "field_schema": dict(dataset.field_schema),
        "annotation_fields": list(dataset.annotation_fields),
        "annotation_field_summaries": dict(getattr(dataset, "annotation_field_summaries", {})),
        "segmentation_usable": True,
        "official_splits_used": False,
        "split_version": split_document["split_version"],
        "split_manifest": split_document,
        "split_manifest_sha256": _canonical_hash(split_document),
        "source_dataset_fingerprint": split_document["dataset_fingerprint"],
        "category_mapping": enriched_mapping,
        "annotation_hashes": {},
        "splits": {},
        "prepared_files": {},
        "license_provenance": {
            "repository_file": "CarDD_license.pdf",
            "conditions": "The Hugging Face mirror does not supersede the original image licensing conditions. "
            "Images are not redistributed in the Git repository.",
        },
        "metrics_scope": "Metrics are evaluated on the project's deterministic held-out split of the "
        f"{len(dataset.samples):,}-sample Hugging Face CarDD mirror, not on the official CarDD test split.",
    }
    return classes, category_mapping, manifest


def _hf_categories(dataset):
    from vision.taxonomy import map_category_name

    classes = sorted(dataset.classes)
    category_mapping = [
        {"category_id": index + 1, "name": name, "yolo_class_id": index} for index, name in enumerate(classes)
    ]
    try:
        enriched_mapping = [dict(item, damage_type=map_category_name(item["name"]).value) for item in category_mapping]
    except ValueError as exc:
        raise DatasetValidationError(f"Unsupported domain taxonomy category: {exc}") from exc
    if len({item["damage_type"] for item in enriched_mapping}) != len(enriched_mapping):
        raise DatasetValidationError("HF classes collapse to duplicate domain damage types")
    return classes, category_mapping, enriched_mapping
