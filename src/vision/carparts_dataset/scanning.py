"""Per-split scanning of Carparts-Seg images, labels and hashes."""

import hashlib
from collections import Counter

from vision.carparts_dataset.constants import IMAGE_SUFFIXES
from vision.dataset import DatasetValidationError, validate_segmentation_label
from vision.training import file_sha256


def _inspect_split(report, split, context):
    document, root, mapping, seen_images = context
    expected_path = f"images/{split}"
    if document.get(split) != expected_path:
        raise DatasetValidationError(f"Carparts-Seg must retain its supplied {expected_path} split")
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        report["split_path_errors"].append(f"Missing {split} image/label directories")
    images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    state = {
        "instances": Counter(),
        "class_images": Counter(),
        "digest": hashlib.sha256(),
        "entries": [],
        "expected_labels": set(),
        "empty_images": [],
    }
    for image in images:
        _inspect_image(report, image, (image_dir, label_dir, root, mapping, seen_images), state)
    _record_split(report, split, images, (label_dir, root, mapping), state)


def _inspect_image(report, image, dirs, state):
    from PIL import Image

    image_dir, label_dir, root, mapping, seen_images = dirs
    resolved = image.resolve()
    if not resolved.is_relative_to(root) or resolved in seen_images:
        raise DatasetValidationError(f"Image escapes dataset root or occurs in multiple splits: {image}")
    seen_images.add(resolved)
    label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
    state["expected_labels"].add(label.resolve())
    width = height = None
    try:
        with Image.open(image) as decoded:
            width, height = decoded.size
            decoded.verify()
        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions")
    except (OSError, ValueError) as exc:
        report["invalid_images"].append({"image": str(image.relative_to(root)), "error": str(exc)})
    rows = []
    if not label.is_file():
        report["missing_labels"].append(str(label.relative_to(root)))
    else:
        try:
            rows = validate_segmentation_label(label, class_count=len(mapping))
        except DatasetValidationError as exc:
            report["invalid_labels"].append({"label": str(label.relative_to(root)), "error": str(exc)})
        state["digest"].update(label.relative_to(root).as_posix().encode())
        state["digest"].update(bytes.fromhex(file_sha256(label)))
    if not rows:
        state["empty_images"].append(str(image.relative_to(root)))
    state["instances"].update(row[0] for row in rows)
    state["class_images"].update({row[0] for row in rows})
    state["entries"].append(_image_entry(image, label, root, (width, height), rows))


def _image_entry(image, label, root, size, rows):
    width, height = size
    return {
        "image": image.relative_to(root).as_posix(),
        "label": label.relative_to(root).as_posix(),
        "width": width,
        "height": height,
        "row_count": len(rows),
    }


def _record_split(report, split, images, dirs, state):
    label_dir, root, mapping = dirs
    instances, class_images = state["instances"], state["class_images"]
    labels = {path.resolve() for path in label_dir.rglob("*.txt")}
    report["orphan_labels"].extend(str(path.relative_to(root)) for path in sorted(labels - state["expected_labels"]))
    report["splits"][split] = {
        "image_count": len(images),
        "label_count": len(labels),
        "annotation_count": sum(instances.values()),
        "images_without_annotations": state["empty_images"],
        "images_without_annotation_count": len(state["empty_images"]),
        "annotations_per_category": {item["name"]: instances[item["yolo_class_id"]] for item in mapping},
        "images_per_category": {item["name"]: class_images[item["yolo_class_id"]] for item in mapping},
    }
    report["prepared_files"][split] = state["entries"]
    report["annotation_hashes"][split] = state["digest"].hexdigest()
