"""Annotations."""

from __future__ import annotations

from collections import Counter

from vision.dataset._models import DatasetValidationError
from vision.dataset._polygons import _polygon


def _convert_coco_annotations(conversion, split, images):
    rows = {image_id: [] for image_id in images}
    normalized_annotations = []
    skipped: Counter[str] = Counter()
    skipped_by_class: Counter[str] = Counter()
    stitched_multipart_count = 0
    class_names = {item["category_id"]: item["name"] for item in conversion.manifest["category_mapping"]}
    for annotation in split.document["annotations"]:
        image = images[annotation["image_id"]]
        polygon, reason = _polygon(annotation, image["width"], image["height"])
        class_id = conversion.class_ids[annotation["category_id"]]
        if reason is None and (class_id, polygon) in rows[annotation["image_id"]]:
            reason = "duplicate_polygon"
        if reason:
            skipped[reason] += 1
            skipped_by_class[class_names[annotation["category_id"]]] += 1
            if conversion.unsupported == "error":
                raise DatasetValidationError(
                    f"{split.name} annotation {annotation['id']}: unusable segmentation ({reason}). "
                    "Fix the source annotations or explicitly choose --conversion.unsupported skip to measure "
                    "and exclude it; no bounding-box fallback."
                )
            continue
        stitched_multipart_count += len(annotation.get("segmentation", [])) > 1
        rows[annotation["image_id"]].append((class_id, polygon or []))
        normalized_annotations.append(_normalized_coco_annotation(annotation, class_id, polygon, image))
    return rows, normalized_annotations, skipped, skipped_by_class, stitched_multipart_count


def _normalized_coco_annotation(annotation, class_id, polygon, image):
    xs = (polygon or [])[::2]
    ys = (polygon or [])[1::2]
    return dict(
        annotation,
        category_id=class_id + 1,
        iscrowd=0,
        segmentation=[
            [value * (image["width"] if i % 2 == 0 else image["height"]) for i, value in enumerate(polygon or [])]
        ],
        bbox=[
            min(xs) * image["width"],
            min(ys) * image["height"],
            (max(xs) - min(xs)) * image["width"],
            (max(ys) - min(ys)) * image["height"],
        ],
    )
