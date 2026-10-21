"""Test identity."""

import pytest

from tests.intelligence.vision.datasets.coco._support import edit_split, synthetic_cardd
from vision.dataset import (
    DatasetValidationError,
    discover_cardd_dataset,
    read_category_mapping,
)


def test_image_ids_cannot_overlap_across_splits(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "test", lambda doc: doc["images"][0].update(id=1))
    with pytest.raises(DatasetValidationError, match="Split leakage"):
        discover_cardd_dataset(root)


def test_filenames_cannot_overlap_across_splits(tmp_path):
    root = synthetic_cardd(tmp_path)
    (root / "test2017" / "synthetic_test.png").rename(root / "test2017" / "synthetic_train.png")
    edit_split(root, "test", lambda doc: doc["images"][0].update(file_name="synthetic_train.png"))
    with pytest.raises(DatasetValidationError, match="Split leakage"):
        discover_cardd_dataset(root)


def test_category_mapping_must_agree_across_splits(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "val", lambda doc: doc["categories"][0].update(name="crack"))
    with pytest.raises(DatasetValidationError, match="differ between splits"):
        discover_cardd_dataset(root)


@pytest.mark.parametrize(
    "categories",
    [
        [],
        [{"id": 2, "name": "dent"}, {"id": 2, "name": "scratch"}],
        [{"id": 2, "name": "dent"}, {"id": 3, "name": "DENT"}],
        [{"id": True, "name": "dent"}],
        [{"id": 2, "name": ""}],
    ],
)
def test_invalid_category_mapping_rejected(categories):
    with pytest.raises(DatasetValidationError):
        read_category_mapping({"categories": categories})
