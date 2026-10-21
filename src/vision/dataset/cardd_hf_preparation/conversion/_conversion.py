"""Conversion."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from vision.dataset import (
    SPLITS,
    DatasetValidationError,
    _check_image,
    annotation_hash,
)
from vision.dataset.cardd_hf_preparation._fingerprint import _canonical_hash
from vision.dataset.cardd_hf_preparation.conversion._context import (
    _HfConversion,
    _HfSplitInventory,
)
from vision.dataset.cardd_hf_preparation.conversion._files import _write_hf_sample
from vision.dataset.cardd_hf_preparation.conversion._geometry import _hf_annotation, _hf_polygon


def _convert_hf_splits(dataset, split_document, manifest, stage):
    conversion = _HfConversion(dataset, split_document, manifest)
    documents, expected = {}, {}
    for split in SPLITS:
        documents[split], expected[split] = _convert_hf_split(conversion, split, stage)
    return documents, expected


def _convert_hf_split(conversion, split, stage):
    inventory = _HfSplitInventory()
    for sample_id in conversion.split_document["sample_ids"][split]:
        _convert_hf_sample(conversion, inventory, sample_id, split, stage)
    document = {
        "images": inventory.images,
        "annotations": inventory.annotations,
        "categories": [{"id": item["category_id"], "name": item["name"]} for item in conversion.category_mapping],
    }
    conversion.manifest["annotation_hashes"][split] = _canonical_hash(document)
    conversion.manifest["prepared_files"][split] = inventory.entries
    per_category = {name: inventory.annotation_counts[name] for name in conversion.classes}
    per_class_images = {name: inventory.image_counts[name] for name in conversion.classes}
    unlabeled = sum(entry["row_count"] == 0 for entry in inventory.entries)
    conversion.manifest["splits"][split] = {
        "image_count": len(inventory.entries),
        "annotation_count": len(inventory.annotations),
        "annotations_per_category": per_category,
        "images_per_category": per_class_images,
        "images_without_annotations": unlabeled,
        "converted_annotation_count": len(inventory.annotations),
        "skipped_annotations": {},
        "skipped_annotations_per_category": {},
        "stitched_multipart_annotations": inventory.stitched_count,
        "source_mask_holes": inventory.hole_count,
        "label_count": len(inventory.entries),
        "images_without_converted_annotations": unlabeled,
        "converted_annotations_per_category": per_category,
        "converted_images_per_category": per_class_images,
    }
    return document, inventory.expected


def _convert_hf_sample(conversion, inventory, sample_id, split, stage):
    sample = conversion.sample_lookup[sample_id]
    _check_image(Path(sample.filepath), sample.width, sample.height)
    if annotation_hash(Path(sample.filepath)) != sample.image_sha256:
        raise DatasetValidationError(f"Source image changed since HF inspection: {sample.sample_id}")
    filename = hashlib.sha256(sample_id.encode()).hexdigest() + Path(sample.filepath).suffix.lower()
    image_id = conversion.image_ids[sample_id]
    image = {"id": image_id, "file_name": filename, "width": sample.width, "height": sample.height}
    inventory.images.append(image)
    rows = _hf_rows(conversion, inventory, sample, image_id)
    inventory.image_counts.update({instance.label for instance in sample.instances})
    image_relative, label_relative = _write_hf_sample(stage, split, sample, filename, rows)
    inventory.expected[filename] = rows
    inventory.entries.append(
        {
            "image": image_relative,
            "label": label_relative,
            "image_id": image_id,
            "source_sample_id": sample_id,
            "image_sha256": sample.image_sha256,
            "width": sample.width,
            "height": sample.height,
            "row_count": len(rows),
            "class_counts": dict(Counter(str(row[0]) for row in rows)),
        }
    )


def _hf_rows(conversion, inventory, sample, image_id):
    rows = []
    for instance_index, instance in enumerate(sample.instances):
        parts, polygon = _hf_polygon(sample, instance, instance_index, conversion)
        class_id = conversion.class_ids[instance.label]
        if (class_id, polygon) in rows:
            raise DatasetValidationError(
                f"HF sample {sample.sample_id}: duplicate polygon would be silently removed by YOLO"
            )
        rows.append((class_id, polygon))
        inventory.stitched_count += len(parts) > 1
        inventory.hole_count += getattr(instance, "mask_hole_count", 0)
        inventory.annotation_counts[instance.label] += 1
        _hf_annotation(sample, polygon, image_id, class_id, inventory.annotations)
    return rows
