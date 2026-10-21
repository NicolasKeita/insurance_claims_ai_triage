"""Test discovery."""

import pytest

from tests.intelligence.vision.datasets.coco._support import synthetic_cardd
from vision.dataset import (
    DatasetValidationError,
    discover_cardd_dataset,
    inspect_cardd_dataset,
)


def test_missing_environment_has_manual_download_instruction(monkeypatch):
    monkeypatch.delenv("CARDD_DATASET_DIR", raising=False)
    with pytest.raises(DatasetValidationError, match="set CARDD_DATASET_DIR"):
        discover_cardd_dataset()


def test_missing_directory_is_explicit(tmp_path):
    with pytest.raises(DatasetValidationError, match="does not exist"):
        discover_cardd_dataset(tmp_path / "missing")


def test_discovery_reads_real_categories_and_split_statistics(tmp_path, monkeypatch):
    root = synthetic_cardd(tmp_path)
    monkeypatch.setenv("CARDD_DATASET_DIR", str(root))
    summary = inspect_cardd_dataset(discover_cardd_dataset())
    assert summary["category_mapping"] == [
        {"category_id": 7, "name": "dent", "yolo_class_id": 0},
        {"category_id": 42, "name": "scratch", "yolo_class_id": 1},
    ]
    assert summary["splits"]["train"] == {
        "image_count": 1,
        "annotation_count": 1,
        "annotations_per_category": {"dent": 0, "scratch": 1},
        "images_per_category": {"dent": 0, "scratch": 1},
        "images_without_annotations": 0,
    }


@pytest.mark.parametrize("split,remove", [("train", "annotation"), ("val", "images"), ("test", "annotation")])
def test_requires_all_official_splits(tmp_path, split, remove):
    root = synthetic_cardd(tmp_path)
    if remove == "annotation":
        (root / "annotations" / f"instances_{split}2017.json").unlink()
    else:
        (root / f"{split}2017" / f"synthetic_{split}.png").unlink()
        (root / f"{split}2017").rmdir()
    with pytest.raises(DatasetValidationError, match=split):
        discover_cardd_dataset(root)
