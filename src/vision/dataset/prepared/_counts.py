from __future__ import annotations

from vision.dataset._models import DatasetValidationError


def _prepared_split_counts(manifest, split, inventory, mapping):
    row_count = inventory.row_count
    actual_class_counts = inventory.class_counts
    actual_class_images = inventory.class_images
    counts = manifest["splits"][split]
    if (
        counts.get("image_count") != inventory.image_count
        or counts.get("label_count") != inventory.image_count
        or counts.get("converted_annotation_count") != row_count
    ):
        raise DatasetValidationError(f"Prepared {split} counts disagree with actual labels/images")
    skipped = counts.get("skipped_annotations", {})
    if (
        not isinstance(skipped, dict)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in skipped.values())
        or counts.get("annotation_count") != row_count + sum(skipped.values())
    ):
        raise DatasetValidationError(f"Prepared {split} annotation/skip counts are inconsistent")
    if counts.get("converted_annotations_per_category") != {
        item["name"]: actual_class_counts[item["yolo_class_id"]] for item in mapping
    } or counts.get("converted_images_per_category") != {
        item["name"]: actual_class_images[item["yolo_class_id"]] for item in mapping
    }:
        raise DatasetValidationError(f"Prepared {split} category distribution disagrees with actual labels")
