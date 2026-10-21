from __future__ import annotations

import os
from pathlib import Path

from vision.dataset._models import DatasetValidationError


def _discover_path(root: Path, candidates: list[str], kind: str, split: str) -> Path:
    found = [root / candidate for candidate in candidates if (root / candidate).exists()]
    if len(found) != 1:
        detail = "none found" if not found else "ambiguous: " + ", ".join(str(p) for p in found)
        raise DatasetValidationError(
            f"CarDD {split} {kind}: {detail}. Expected one of {candidates}; "
            "use explicit path mappings for a custom layout."
        )
    return found[0].resolve()


def _split_paths(root, split, annotation_files, image_directories):
    if annotation_files is None:
        annotation_path = _discover_path(
            root,
            [f"annotations/instances_{split}2017.json", f"annotations/instances_{split}.json"],
            "annotations",
            split,
        )
    else:
        annotation_path = (root / annotation_files[split]).resolve()
    if not annotation_path.is_file():
        raise DatasetValidationError(f"CarDD {split} annotation file is missing: {annotation_path}")
    if image_directories is None:
        image_dir = _discover_path(
            root, [f"{split}2017", f"images/{split}2017", f"images/{split}", split], "image directory", split
        )
    else:
        image_dir = (root / image_directories[split]).resolve()
    if not image_dir.is_dir():
        raise DatasetValidationError(f"CarDD {split} image directory is missing: {image_dir}")
    return annotation_path, image_dir


def _dataset_root(dataset_dir):
    configured = dataset_dir if dataset_dir is not None else os.environ.get("CARDD_DATASET_DIR")
    if not configured:
        raise DatasetValidationError(
            "CarDD dataset unavailable. Obtain CarDD manually from its official site under its conditions, "
            "then set CARDD_DATASET_DIR to the CarDD_COCO directory. No automatic download is performed."
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise DatasetValidationError(
            f"CARDD_DATASET_DIR does not exist or is not a directory: {root}. "
            "Set it to the manually obtained CarDD_COCO directory."
        )
    return root
