"""Inventory."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from vision.carparts_dataset._constants import IMAGE_SUFFIXES
from vision.carparts_dataset._context import _Inspection
from vision.dataset import SPLITS, DatasetValidationError, validate_segmentation_label
from vision.training import file_sha256


def _inspect_carparts_splits(document, root, mapping, report):
    inspection = _Inspection(root, mapping, report)
    for split in SPLITS:
        _inspect_carparts_split(inspection, split, document)


def _inspect_carparts_split(inspection, split, document):
    expected_path = f"images/{split}"
    if document.get(split) != expected_path:
        raise DatasetValidationError(f"Carparts-Seg must retain its supplied {expected_path} split")
    image_dir, label_dir = inspection.root / "images" / split, inspection.root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        inspection.report["split_path_errors"].append(f"Missing {split} image/label directories")
    images, instances, class_images, digest, entries, expected_labels, empty_images = _inspect_carparts_images(
        inspection, image_dir, label_dir
    )
    labels = {path.resolve() for path in label_dir.rglob("*.txt")}
    inspection.report["orphan_labels"].extend(
        str(path.relative_to(inspection.root)) for path in sorted(labels - expected_labels)
    )
    inspection.report["splits"][split] = {
        "image_count": len(images),
        "label_count": len(labels),
        "annotation_count": sum(instances.values()),
        "images_without_annotations": empty_images,
        "images_without_annotation_count": len(empty_images),
        "annotations_per_category": {item["name"]: instances[item["yolo_class_id"]] for item in inspection.mapping},
        "images_per_category": {item["name"]: class_images[item["yolo_class_id"]] for item in inspection.mapping},
    }
    inspection.report["prepared_files"][split] = entries
    inspection.report["annotation_hashes"][split] = digest.hexdigest()


def _inspect_carparts_image(inspection, image, label, digest):
    from PIL import Image

    width = height = None
    try:
        with Image.open(image) as decoded:
            width, height = decoded.size
            decoded.verify()
        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions")
    except (OSError, ValueError) as exc:
        inspection.report["invalid_images"].append(
            {"image": str(image.relative_to(inspection.root)), "error": str(exc)}
        )
    rows = []
    if not label.is_file():
        inspection.report["missing_labels"].append(str(label.relative_to(inspection.root)))
    else:
        try:
            rows = validate_segmentation_label(label, class_count=len(inspection.mapping))
        except DatasetValidationError as exc:
            inspection.report["invalid_labels"].append(
                {"label": str(label.relative_to(inspection.root)), "error": str(exc)}
            )
        digest.update(label.relative_to(inspection.root).as_posix().encode())
        digest.update(bytes.fromhex(file_sha256(label)))
    return width, height, rows


def _inspect_carparts_images(inspection, image_dir, label_dir):
    images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    instances: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    digest = hashlib.sha256()
    entries = []
    expected_labels: set[Path] = set()
    empty_images = []
    for image in images:
        resolved = image.resolve()
        if not resolved.is_relative_to(inspection.root) or resolved in inspection.seen_images:
            raise DatasetValidationError(f"Image escapes dataset root or occurs in multiple splits: {image}")
        inspection.seen_images.add(resolved)
        label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
        expected_labels.add(label.resolve())
        width, height, rows = _inspect_carparts_image(inspection, image, label, digest)
        if not rows:
            empty_images.append(str(image.relative_to(inspection.root)))
        instances.update(row[0] for row in rows)
        class_images.update({row[0] for row in rows})
        entries.append(
            {
                "image": image.relative_to(inspection.root).as_posix(),
                "label": label.relative_to(inspection.root).as_posix(),
                "width": width,
                "height": height,
                "row_count": len(rows),
            }
        )
    return images, instances, class_images, digest, entries, expected_labels, empty_images
