"""Validation."""

from __future__ import annotations

from typing import Any

from vision.cardd_hf_preparation._contracts import (
    _hf_source_contract,
    _hf_source_lookup,
    _validate_hf_distribution,
)
from vision.dataset import (
    SPLITS,
    DatasetValidationError,
)


def validate_hf_prepared_manifest(manifest: dict[str, Any]) -> None:
    """Validate mirror provenance without loading original data or FiftyOne."""
    split_document, mapping = _hf_source_contract(manifest)
    sample_ids = split_document.get("sample_ids")
    if not isinstance(sample_ids, dict) or set(sample_ids) != set(SPLITS):
        raise DatasetValidationError("Prepared HF frozen split must contain train/val/test sample IDs")
    source_lookup = _hf_source_lookup(split_document, manifest)
    label_ids = {item["name"]: str(item["yolo_class_id"]) for item in mapping}
    all_ids: set[str] = set()
    hash_splits: dict[str, str] = {}
    count = 0
    prepared_files = manifest.get("prepared_files", {})
    for split in SPLITS:
        ids = sample_ids[split]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
            or all_ids.intersection(ids)
        ):
            raise DatasetValidationError(
                f"Prepared HF frozen split contains invalid or overlapping sample IDs: {split}"
            )
        entries = prepared_files.get(split)
        if (
            not isinstance(entries, list)
            or any(not isinstance(entry, dict) for entry in entries)
            or len(entries) != len(ids)
            or {entry.get("source_sample_id") for entry in entries} != set(ids)
        ):
            raise DatasetValidationError(f"Prepared HF images disagree with frozen split membership: {split}")
        for entry in entries:
            _validate_hf_entry(entry, source_lookup, label_ids, hash_splits, split)
        _validate_hf_distribution(manifest, split_document, split)
        all_ids.update(ids)
        count += len(ids)
    if manifest.get("actual_sample_count") != count or all_ids != set(source_lookup):
        raise DatasetValidationError("Prepared HF actual sample count disagrees with frozen split")


def _validate_hf_entry(entry, source_lookup, label_ids, hash_splits, split):
    source = source_lookup.get(entry["source_sample_id"])
    if source is None or any(
        entry.get(entry_key) != source.get(source_key)
        for entry_key, source_key in (
            ("image_sha256", "image_sha256"),
            ("width", "width"),
            ("height", "height"),
            ("row_count", "annotation_count"),
        )
    ):
        raise DatasetValidationError("Prepared HF image/instance metadata disagrees with frozen source sample")
    per_category = source.get("annotations_per_category")
    if (
        not isinstance(per_category, dict)
        or any(name not in label_ids for name in per_category)
        or entry.get("class_counts") != {label_ids[name]: count for name, count in per_category.items()}
    ):
        raise DatasetValidationError("Prepared HF class counts disagree with frozen source sample")
    digest = entry.get("image_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise DatasetValidationError("Prepared HF images require SHA-256 content hashes")
    if digest in hash_splits and hash_splits[digest] != split:
        raise DatasetValidationError("Prepared HF split leakage: identical image bytes in multiple splits")
    hash_splits[digest] = split
