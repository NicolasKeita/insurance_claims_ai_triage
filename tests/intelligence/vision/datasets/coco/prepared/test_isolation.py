"""Test isolation."""

import pytest

from tests.intelligence.vision.datasets.coco._support import (
    edit_split,
    prepare_fixture,
    synthetic_cardd,
)
from vision.dataset import (
    DatasetValidationError,
    discover_cardd_dataset,
    prepare_cardd_yolo,
)


def test_annotation_image_dimensions_must_match_real_pixels(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["images"][0].update(width=21))
    with pytest.raises(DatasetValidationError, match="dimensions disagree"):
        prepare_cardd_yolo(discover_cardd_dataset(root), tmp_path / "prepared", manifest_path=None, converter="native")


def test_preparation_never_overwrites_or_writes_sources(tmp_path):
    root, output, _ = prepare_fixture(tmp_path)
    dataset = discover_cardd_dataset(root)
    with pytest.raises(DatasetValidationError, match="already exists"):
        prepare_cardd_yolo(dataset, output, manifest_path=None, converter="native")
    with pytest.raises(DatasetValidationError, match="outside and separate"):
        prepare_cardd_yolo(dataset, root / "prepared", manifest_path=None, converter="native")
    with pytest.raises(DatasetValidationError, match="manifest must be outside"):
        prepare_cardd_yolo(dataset, tmp_path / "another", manifest_path=root / "manifest.json", converter="native")


def test_source_path_traversal_is_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["images"][0].update(file_name="../synthetic_train.png"))
    with pytest.raises(DatasetValidationError, match="Unsafe relative filename"):
        discover_cardd_dataset(root)
