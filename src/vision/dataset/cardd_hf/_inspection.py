"""Inspection."""

from __future__ import annotations

from collections import Counter
from typing import Any

from vision.dataset.cardd_hf._models import HfDamageDataset


def inspect_cardd_hf(dataset: HfDamageDataset) -> dict[str, Any]:
    counts = Counter(instance.label for sample in dataset.samples for instance in sample.instances)
    image_counts = Counter(
        label for sample in dataset.samples for label in {instance.label for instance in sample.instances}
    )
    filenames = Counter(sample.filepath.name.casefold() for sample in dataset.samples)
    paths = Counter(str(sample.filepath.resolve()).casefold() for sample in dataset.samples)
    hashes = Counter(sample.image_sha256 for sample in dataset.samples)
    widths, heights = [s.width for s in dataset.samples], [s.height for s in dataset.samples]
    return {
        "source_name": f"CarDD HF mirror ({dataset.repository})",
        "source_repo": dataset.repository,
        "source_revision": dataset.revision,
        "source_format": dataset.source_format,
        "official_cardd_splits_used": False,
        "sample_count": len(dataset.samples),
        "field_schema": dataset.field_schema,
        "annotation_fields": list(dataset.annotation_fields),
        "annotation_field_summaries": dataset.annotation_field_summaries,
        "damage_classes": list(dataset.classes),
        "labeled_instance_count": sum(counts.values()),
        "instances_per_class": {name: counts[name] for name in dataset.classes},
        "images_per_class": {name: image_counts[name] for name in dataset.classes},
        "images_without_annotations": sum(not sample.instances for sample in dataset.samples),
        "segmentation_usable": bool(counts),
        "segmentation": _segmentation_metadata(dataset),
        "image_dimensions": _image_dimensions(widths, heights),
        "duplicate_checks": {
            "duplicate_filenames": sorted(k for k, v in filenames.items() if v > 1),
            "duplicate_filepaths": sorted(k for k, v in paths.items() if v > 1),
            "duplicate_image_sha256": sorted(k for k, v in hashes.items() if v > 1),
            "image_hash_scope": "SHA-256 of full encoded image bytes; detects exact copies even when renamed",
        },
        "sample_id_policy": "source coco_id when available, otherwise filename and full image SHA-256; "
        "local FiftyOne/MongoDB IDs are not used",
    }


def _segmentation_metadata(dataset):
    return {
        "instance_masks": sum(i.source_kind == "instance_mask" for s in dataset.samples for i in s.instances),
        "polylines": sum(i.source_kind == "polyline" for s in dataset.samples for i in s.instances),
        "multipart_instances": sum(len(i.polygons) > 1 for s in dataset.samples for i in s.instances),
        "mask_holes": sum(i.mask_hole_count for s in dataset.samples for i in s.instances),
        "bbox_fallback": False,
        "mask_polygon_policy": "trace binary pixel-cell edges with 8-connected foreground; "
        "preserve component and hole vertices with retraced hole bridges",
        "representation_limitation": "YOLO has one contour per instance; "
        "multipart/hole bridges may add pixels during rasterization",
    }


def _image_dimensions(widths, heights):
    return {
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "distinct_sizes": len(set(zip(widths, heights))),
    }
