from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

from vision.dataset._files import (
    _check_image,
    _contained_file,
    _integer,
    _relative_filename,
    annotation_hash,
)
from vision.dataset._models import DatasetValidationError
from vision.dataset._polygons import validate_segmentation_label


def _prepared_entry(entry, dataset_root, split, mapping, is_hf_mirror):
    try:
        image_name = _relative_filename(entry["image"])
        label_name = _relative_filename(entry["label"])
        image_id = _integer(entry["image_id"], "Prepared image ID", minimum=0)
        width = _integer(entry["width"], "Prepared image width", minimum=1)
        height = _integer(entry["height"], "Prepared image height", minimum=1)
        count = _integer(entry["row_count"], "Prepared label count", minimum=0)
    except (KeyError, TypeError) as exc:
        raise DatasetValidationError("Invalid prepared image/label metadata") from exc
    if (
        not image_name.startswith(f"images/{split}/")
        or not label_name.startswith(f"labels/{split}/")
        or PurePosixPath(image_name).with_suffix(".txt").parts[2:] != PurePosixPath(label_name).parts[2:]
    ):
        raise DatasetValidationError("Prepared label does not reference a corresponding split image")
    image_path = _contained_file(dataset_root, image_name)
    label_path = _contained_file(dataset_root, label_name)
    _check_image(image_path, width, height)
    if is_hf_mirror and annotation_hash(image_path) != entry.get("image_sha256"):
        raise DatasetValidationError(f"Prepared HF image content changed: {image_path}")
    rows = validate_segmentation_label(label_path, class_count=len(mapping), expected_count=count)
    if dict(Counter(str(row[0]) for row in rows)) != entry.get("class_counts") or annotation_hash(
        label_path
    ) != entry.get("label_sha256"):
        raise DatasetValidationError(f"Prepared labels changed or class counts disagree with manifest: {label_path}")
    return image_name, label_name, image_id, image_path, label_path, rows
