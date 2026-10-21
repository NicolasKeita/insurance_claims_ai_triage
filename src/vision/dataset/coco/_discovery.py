"""Discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.dataset._constants import SPLITS
from vision.dataset._files import _integer, _read_json
from vision.dataset._models import CarddDataset, CocoSplit, DatasetValidationError
from vision.dataset.coco._locations import _dataset_root, _split_paths
from vision.dataset.coco._references import _annotation_references, _image_identities


def read_category_mapping(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Map actual COCO category IDs, including noncontiguous IDs, to YOLO IDs."""
    categories = document.get("categories")
    if not isinstance(categories, list) or not categories:
        raise DatasetValidationError("COCO categories must be a nonempty list")
    names: set[str] = set()
    ids: set[int] = set()
    result = []
    for category in categories:
        if not isinstance(category, dict):
            raise DatasetValidationError("Each COCO category must be an object")
        category_id = _integer(category.get("id"), "COCO category ID", minimum=0)
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DatasetValidationError("COCO category names must be nonempty strings")
        if category_id in ids or name.casefold().strip() in names:
            raise DatasetValidationError("Duplicate COCO category ID or name")
        ids.add(category_id)
        names.add(name.casefold().strip())
        result.append({"category_id": category_id, "name": name})
    result.sort(key=lambda item: item["category_id"])
    return [dict(item, yolo_class_id=index) for index, item in enumerate(result)]


def discover_cardd_dataset(
    dataset_dir: Path | str | None = None,
    *,
    annotation_files: dict[str, Path | str] | None = None,
    image_directories: dict[str, Path | str] | None = None,
) -> CarddDataset:
    """Read official CarDD_COCO layout, require all splits, and reject leakage."""
    root = _dataset_root(dataset_dir)
    for mapping in (annotation_files, image_directories):
        if mapping is not None and set(mapping) != set(SPLITS):
            raise DatasetValidationError("Explicit split paths must define exactly train, val and test")
    splits = {}
    category_mapping = None
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for split in SPLITS:
        annotation_path, image_dir = _split_paths(root, split, annotation_files, image_directories)
        document = _read_json(annotation_path)
        mapping = read_category_mapping(document)
        if category_mapping is None:
            category_mapping = mapping
        elif category_mapping != mapping:
            raise DatasetValidationError(f"COCO category IDs/names differ between splits: {split}")
        images = document.get("images")
        annotations = document.get("annotations")
        if not isinstance(images, list) or not images or not isinstance(annotations, list):
            raise DatasetValidationError(f"{split}: COCO images must be nonempty and annotations must be a list")
        split_ids, split_names = _image_identities(images, split, image_dir)
        if split_ids & seen_ids or split_names & seen_names:
            raise DatasetValidationError(f"Split leakage: {split} overlaps previous splits by image ID or filename")
        seen_ids.update(split_ids)
        seen_names.update(split_names)
        _annotation_references(annotations, split, mapping, split_ids)
        splits[split] = CocoSplit(split, annotation_path, image_dir, document)
    return CarddDataset(root, splits, category_mapping or [])
