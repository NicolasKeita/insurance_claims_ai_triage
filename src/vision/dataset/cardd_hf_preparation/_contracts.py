"""Contracts."""

from __future__ import annotations

from vision.dataset import (
    DatasetValidationError,
)
from vision.dataset.cardd_hf_preparation._fingerprint import _canonical_hash


def _hf_source_contract(manifest):
    split_document = manifest.get("split_manifest")
    if not isinstance(split_document, dict) or _canonical_hash(split_document) != manifest.get("split_manifest_sha256"):
        raise DatasetValidationError("Prepared HF frozen split manifest hash disagrees")
    if manifest.get("source_name") != "CarDD HF mirror" or manifest.get("official_splits_used") is not False:
        raise DatasetValidationError("Prepared HF dataset must explicitly identify the mirror and custom splits")
    for source_key, split_key in (
        ("source_repository", "source_repo"),
        ("source_revision", "source_revision"),
        ("actual_sample_count", "sample_count"),
        ("source_dataset_fingerprint", "dataset_fingerprint"),
        ("split_version", "split_version"),
    ):
        if manifest.get(source_key) != split_document.get(split_key):
            raise DatasetValidationError(f"Prepared HF provenance disagrees with frozen split: {source_key}")
    if not isinstance(manifest.get("source_repository"), str) or not manifest["source_repository"].strip():
        raise DatasetValidationError("Prepared HF source repository is missing")
    fields = manifest.get("annotation_fields")
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields):
        raise DatasetValidationError("Prepared HF annotation fields are missing")
    if (
        fields != split_document.get("annotation_fields")
        or manifest.get("field_schema") != split_document.get("field_schema")
        or manifest.get("source_format") != split_document.get("source_format")
    ):
        raise DatasetValidationError("Prepared HF annotation fields/schema disagree with frozen source")
    if manifest.get("segmentation_usable") is not True:
        raise DatasetValidationError("Prepared HF dataset requires usable instance segmentation")
    mapping = manifest.get("category_mapping", [])
    if manifest.get("actual_damage_classes") != [item.get("name") for item in mapping]:
        raise DatasetValidationError("Prepared HF damage classes disagree with category mapping")
    if split_document.get("category_mapping") != [
        {key: item.get(key) for key in ("category_id", "name", "yolo_class_id")} for item in mapping
    ]:
        raise DatasetValidationError("Prepared HF category mapping disagrees with frozen split")
    return split_document, mapping


def _hf_source_lookup(split_document, manifest):
    source_samples = split_document.get("samples")
    if not isinstance(source_samples, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) for row in source_samples
    ):
        raise DatasetValidationError("Prepared HF frozen split source sample records are invalid")
    source_lookup = {row["sample_id"]: row for row in source_samples}
    if len(source_lookup) != len(source_samples) or len(source_samples) != manifest.get("actual_sample_count"):
        raise DatasetValidationError("Prepared HF frozen source sample IDs/counts are invalid")
    return source_lookup


def _validate_hf_distribution(manifest, split_document, split):
    counts = manifest.get("splits", {}).get(split)
    frozen_counts = split_document.get("splits", {}).get(split)
    if (
        not isinstance(counts, dict)
        or not isinstance(frozen_counts, dict)
        or any(counts.get(key) != value for key, value in frozen_counts.items())
    ):
        raise DatasetValidationError(f"Prepared HF split distribution disagrees with frozen source: {split}")
    if (
        counts.get("skipped_annotations") != {}
        or counts.get("converted_annotation_count") != frozen_counts.get("annotation_count")
        or counts.get("converted_annotations_per_category") != frozen_counts.get("annotations_per_category")
        or counts.get("converted_images_per_category") != frozen_counts.get("images_per_category")
    ):
        raise DatasetValidationError(f"Prepared HF conversion lost instances or changed frozen distributions: {split}")
