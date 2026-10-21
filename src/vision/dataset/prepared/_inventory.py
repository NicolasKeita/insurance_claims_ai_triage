"""Inventory."""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

from vision.dataset._constants import SPLITS
from vision.dataset._models import DatasetValidationError, _PreparedCounts
from vision.dataset.prepared._counts import _prepared_split_counts
from vision.dataset.prepared._entries import _prepared_entry


def _validate_prepared_splits(document, dataset_root, manifest, is_hf_mirror, mapping):
    seen_ids, seen_names = set(), set()
    for split in SPLITS:
        image_dir, label_dir, entries = _prepared_split_layout(document, dataset_root, manifest, split)
        expected_images, expected_labels, split_names, split_ids = set(), set(), set(), set()
        row_count = 0
        actual_class_counts: Counter[int] = Counter()
        actual_class_images: Counter[int] = Counter()
        for entry in entries:
            image_name, label_name, image_id, image_path, label_path, rows = _prepared_entry(
                entry, dataset_root, split, mapping, is_hf_mirror
            )
            filename = PurePosixPath(image_name).relative_to(PurePosixPath("images") / split).as_posix().casefold()
            if (
                image_id in split_ids
                or filename in split_names
                or image_path in expected_images
                or label_path in expected_labels
            ):
                raise DatasetValidationError(f"Duplicate prepared image or label metadata in {split}")
            split_ids.add(image_id)
            split_names.add(filename)
            expected_images.add(image_path)
            expected_labels.add(label_path)
            row_count += len(rows)
            actual_class_counts.update(row[0] for row in rows)
            actual_class_images.update({row[0] for row in rows})
        if seen_ids & split_ids or seen_names & split_names:
            raise DatasetValidationError(f"Prepared split leakage in {split}")
        seen_ids.update(split_ids)
        seen_names.update(split_names)
        _verify_split_files(image_dir, label_dir, expected_images, expected_labels, split)
        _prepared_split_counts(
            manifest, split, _PreparedCounts(len(entries), row_count, actual_class_counts, actual_class_images), mapping
        )


def _verify_split_files(image_dir, label_dir, expected_images, expected_labels, split):
    actual_images = {path.resolve() for path in image_dir.rglob("*") if path.is_file()}
    actual_labels = {path.resolve() for path in label_dir.rglob("*") if path.is_file()}
    if actual_images != expected_images or actual_labels != expected_labels:
        raise DatasetValidationError(f"Unexpected or missing files in prepared {split} split")


def _prepared_split_layout(document, dataset_root, manifest, split):
    if document.get(split) != f"images/{split}":
        raise DatasetValidationError(f"Prepared {split} split path must be images/{split}")
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise DatasetValidationError(f"Prepared {split} image/label directory missing")
    entries = manifest["prepared_files"][split]
    if not isinstance(entries, list) or not entries:
        raise DatasetValidationError(f"Prepared {split} image metadata must be nonempty")
    return image_dir, label_dir, entries
