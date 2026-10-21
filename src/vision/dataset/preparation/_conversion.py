"""Conversion."""

from __future__ import annotations

from collections import Counter

from vision.dataset._models import _CocoConversion, _ConversionCounts
from vision.dataset.preparation._annotations import _convert_coco_annotations
from vision.dataset.preparation._images import _copy_coco_images


def _convert_coco_splits(dataset, manifest, stage, class_ids, unsupported):
    conversion = _CocoConversion(manifest, class_ids, unsupported)
    for split_name, split in dataset.splits.items():
        _convert_coco_split(conversion, split_name, split, stage)
    return conversion.expected, conversion.documents


def _convert_coco_split(conversion, split_name, split, stage):
    images = {image["id"]: image for image in split.document["images"]}
    rows, normalized_annotations, skipped, skipped_by_class, stitched_multipart_count = _convert_coco_annotations(
        conversion, split, images
    )
    prepared_files = _copy_coco_images(conversion, split_name, split, stage, rows)
    conversion.manifest["prepared_files"][split_name] = prepared_files
    _update_coco_counts(
        conversion, split_name, rows, _ConversionCounts(skipped, skipped_by_class, stitched_multipart_count)
    )
    conversion.documents[split_name] = {
        "images": list(images.values()),
        "annotations": normalized_annotations,
        "categories": [
            {"id": item["yolo_class_id"] + 1, "name": item["name"]} for item in conversion.manifest["category_mapping"]
        ],
    }


def _update_coco_counts(conversion, split_name, rows, diagnostics):
    skipped = diagnostics.skipped
    skipped_by_class = diagnostics.skipped_by_class
    stitched_multipart_count = diagnostics.stitched_multipart_count
    counts = conversion.manifest["splits"][split_name]
    counts.update(
        converted_annotation_count=sum(len(value) for value in rows.values()),
        skipped_annotations=dict(skipped),
        skipped_annotations_per_category=dict(skipped_by_class),
        stitched_multipart_annotations=stitched_multipart_count,
        label_count=len(rows),
        images_without_converted_annotations=sum(not value for value in rows.values()),
    )
    converted_counts = Counter(class_id for values in rows.values() for class_id, _ in values)
    converted_images = Counter(class_id for values in rows.values() for class_id in {row[0] for row in values})
    counts["converted_annotations_per_category"] = {
        item["name"]: converted_counts[item["yolo_class_id"]] for item in conversion.manifest["category_mapping"]
    }
    counts["converted_images_per_category"] = {
        item["name"]: converted_images[item["yolo_class_id"]] for item in conversion.manifest["category_mapping"]
    }
