"""Support."""

import json
from pathlib import Path

from PIL import Image

from vision.dataset import (
    discover_cardd_dataset,
    prepare_cardd_yolo,
)


def synthetic_cardd(tmp_path: Path) -> Path:
    """
    Tiny generated pixels with arbitrary polygons, not damage examples.

    Noncontiguous/reversed IDs deliberately avoid category_id - 1.
    """
    root = tmp_path / "external_coco"
    (root / "annotations").mkdir(parents=True)
    for index, split in enumerate(("train", "val", "test"), start=1):
        image_dir = root / f"{split}2017"
        image_dir.mkdir()
        filename = f"synthetic_{split}.png"
        Image.new("RGB", (20, 10), color=(index * 30, 0, 0)).save(image_dir / filename)
        document = {
            "images": [{"id": index, "file_name": filename, "width": 20, "height": 10}],
            "categories": [{"id": 42, "name": "scratch"}, {"id": 7, "name": "dent"}],
            "annotations": [
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 42,
                    "bbox": [2, 1, 10, 5],
                    "area": 50,
                    "iscrowd": 0,
                    "segmentation": [[2, 1, 12, 1, 12, 6, 2, 6]],
                }
            ],
        }
        (root / "annotations" / f"instances_{split}2017.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def edit_split(root: Path, split: str, change) -> None:
    path = root / "annotations" / f"instances_{split}2017.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def prepare_fixture(tmp_path: Path, **kwargs):
    root = synthetic_cardd(tmp_path)
    dataset = discover_cardd_dataset(root)
    output = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(
        dataset, output, manifest_path=tmp_path / "portable_manifest.json", converter="native", **kwargs
    )
    return root, output, manifest
