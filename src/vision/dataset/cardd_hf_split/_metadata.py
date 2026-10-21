"""Metadata."""

from __future__ import annotations

from collections import Counter
from typing import Any

from vision.dataset.cardd_hf_split._constants import SPLITS
from vision.dataset.cardd_hf_split._identity import _digest, _Group


def _split_metadata(context: dict[str, Any], assignments: dict[str, list[str]], groups: list[_Group]) -> dict[str, Any]:
    rows = {row["sample_id"]: row for row in context["samples"]}
    classes = [item["name"] for item in context["category_mapping"]]
    split_counts, hashes = {}, {}
    for name in SPLITS:
        samples = [rows[sample_id] for sample_id in assignments[name]]
        instances = Counter(
            label for row in samples for label, count in row["annotations_per_category"].items() for _ in range(count)
        )
        images = Counter(label for row in samples for label in row["annotations_per_category"])
        split_counts[name] = {
            "image_count": len(samples),
            "annotation_count": sum(row["annotation_count"] for row in samples),
            "annotations_per_category": {label: instances[label] for label in classes},
            "images_per_category": {label: images[label] for label in classes},
            "images_without_annotations": sum(row["annotation_count"] == 0 for row in samples),
        }
        hashes[name] = _digest([{key: value for key, value in row.items() if key != "file_name"} for row in samples])
    warnings = _distribution_warnings(classes, split_counts, groups)
    return {
        "splits": split_counts,
        "class_distribution": {
            name: {
                "images": split_counts[name]["images_per_category"],
                "instances": split_counts[name]["annotations_per_category"],
            }
            for name in SPLITS
        },
        "annotation_hashes": hashes,
        "warnings": warnings,
        "split_overlap_check": "passed: sample IDs, resolved filepaths and image SHA256 are disjoint across splits",
    }


def _distribution_warnings(classes, split_counts, groups):
    warnings = []
    for label in classes:
        independent = sum(label in group.image_counts for group in groups)
        if independent < len(SPLITS):
            warnings.append(
                f"Class {label!r} has only {independent} independent image groups; "
                "representation in all three splits is impossible."
            )
        elif any(not split_counts[name]["images_per_category"][label] for name in SPLITS):
            warnings.append(
                f"Class {label!r} is absent from at least one split: "
                "multilabel overlap, duplicate grouping and exact capacities prevented full coverage."
            )
    return warnings
