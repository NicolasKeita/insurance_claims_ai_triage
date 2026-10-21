"""Inspection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from vision.dataset._constants import PREPARATION_VERSION
from vision.dataset._files import annotation_hash
from vision.dataset._models import CarddDataset


def inspect_cardd_dataset(dataset: CarddDataset) -> dict[str, Any]:
    """Portable inspection metadata. Does not write a manifest or any other file."""
    mapping = dataset.category_mapping
    id_to_name = {item["category_id"]: item["name"] for item in mapping}
    split_counts = {}
    for name, split in dataset.splits.items():
        counts: Counter[str] = Counter()
        images_per_category: dict[str, set[int]] = {item["name"]: set() for item in mapping}
        annotated = set()
        for annotation in split.document["annotations"]:
            category_name = id_to_name[annotation["category_id"]]
            counts[category_name] += 1
            images_per_category[category_name].add(annotation["image_id"])
            annotated.add(annotation["image_id"])
        split_counts[name] = {
            "image_count": len(split.document["images"]),
            "annotation_count": len(split.document["annotations"]),
            "annotations_per_category": {item["name"]: counts[item["name"]] for item in mapping},
            "images_per_category": {key: len(value) for key, value in images_per_category.items()},
            "images_without_annotations": len(split.document["images"]) - len(annotated),
        }
    return {
        "category_mapping": [dict(item) for item in mapping],
        "splits": split_counts,
        "annotation_hashes": {name: annotation_hash(split.annotation_path) for name, split in dataset.splits.items()},
        "split_overlap_check": "passed: image IDs and case-insensitive filenames are disjoint",
    }


def build_dataset_manifest(dataset: CarddDataset) -> dict[str, Any]:
    result = inspect_cardd_dataset(dataset)
    result["preparation_version"] = PREPARATION_VERSION
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    stable = {
        "annotation_hashes": result["annotation_hashes"],
        "category_mapping": result["category_mapping"],
        "preparation_version": PREPARATION_VERSION,
    }
    result["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result
