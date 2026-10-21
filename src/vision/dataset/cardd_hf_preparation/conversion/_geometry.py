"""Geometry."""

from __future__ import annotations

from vision.dataset import (
    DatasetValidationError,
    _polygon,
)


def _hf_polygon(sample, instance, instance_index, conversion):
    if instance.label not in conversion.class_ids:
        raise DatasetValidationError(f"HF instance label absent from inspected conversion.classes: {instance.label}")
    parts = [
        [
            coordinate * (sample.width if index % 2 == 0 else sample.height)
            for index, coordinate in enumerate([value for point in component for value in point])
        ]
        for component in instance.polygons
    ]
    polygon, reason = _polygon({"segmentation": parts}, sample.width, sample.height)
    if reason or polygon is None:
        raise DatasetValidationError(
            f"HF sample {sample.sample_id} instance {instance_index}: unusable segmentation ({reason}); "
            "no bounding-box fallback"
        )
    return parts, polygon


def _hf_annotation(sample, polygon, image_id, class_id, annotations):
    xs, ys = polygon[::2], polygon[1::2]
    annotations.append(
        {
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": class_id + 1,
            "iscrowd": 0,
            "segmentation": [
                [value * (sample.width if index % 2 == 0 else sample.height) for index, value in enumerate(polygon)]
            ],
            "bbox": [
                min(xs) * sample.width,
                min(ys) * sample.height,
                (max(xs) - min(xs)) * sample.width,
                (max(ys) - min(ys)) * sample.height,
            ],
        }
    )
