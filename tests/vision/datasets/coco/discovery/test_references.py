"""Test references."""

import pytest

from tests.vision.datasets.coco._support import edit_split, synthetic_cardd
from vision.dataset import (
    DatasetValidationError,
    discover_cardd_dataset,
)


def test_unknown_image_reference_is_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(image_id=999))
    with pytest.raises(DatasetValidationError, match="unknown image ID"):
        discover_cardd_dataset(root)


def test_unknown_category_reference_is_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(category_id=999))
    with pytest.raises(DatasetValidationError, match="unknown category ID"):
        discover_cardd_dataset(root)


def test_annotation_requires_real_source_image(tmp_path):
    root = synthetic_cardd(tmp_path)
    (root / "train2017" / "synthetic_train.png").unlink()
    with pytest.raises(DatasetValidationError, match="Image file is missing"):
        discover_cardd_dataset(root)
