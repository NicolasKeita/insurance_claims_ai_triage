"""Test labels."""

import json

import pytest

from tests.intelligence.vision.datasets.coco._support import prepare_fixture
from vision.dataset import (
    DatasetValidationError,
    load_prepared_dataset,
    validate_segmentation_label,
)


@pytest.mark.parametrize(
    "row", ["2 0 0 1 0 1 1\n", "0 0 0 1.2 0 1 1\n", "0 0 0 nan 0 1 1\n", "0 0 0 1 1\n", "0 0 0 0.5 0.5 1 1\n"]
)
def test_invalid_segmentation_rows_rejected(tmp_path, row):
    path = tmp_path / "label.txt"
    path.write_text(row)
    with pytest.raises(DatasetValidationError):
        validate_segmentation_label(path, class_count=2)


def test_prepared_label_tampering_is_rejected(tmp_path):
    _, output, _ = prepare_fixture(tmp_path)
    (output / "labels" / "train" / "synthetic_train.txt").write_text("0 0.1 0.1 0.6 0.1 0.6 0.6\n")
    with pytest.raises(DatasetValidationError, match="changed|disagree"):
        load_prepared_dataset(output / "dataset.yaml")


def test_prepared_unknown_or_orphan_files_rejected(tmp_path):
    _, output, _ = prepare_fixture(tmp_path)
    (output / "labels" / "test" / "orphan.txt").write_text("")
    with pytest.raises(DatasetValidationError, match="Unexpected or missing files"):
        load_prepared_dataset(output / "dataset.yaml")


def test_prepared_class_count_mismatch_rejected(tmp_path):
    _, output, _ = prepare_fixture(tmp_path)
    path = output / "dataset.yaml"
    document = json.loads(path.read_text())
    document["nc"] = 90
    path.write_text(json.dumps(document))
    with pytest.raises(DatasetValidationError, match="names/count"):
        load_prepared_dataset(path)
