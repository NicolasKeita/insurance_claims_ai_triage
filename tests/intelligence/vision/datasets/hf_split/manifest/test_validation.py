"""Test validation."""

import pytest

from tests.intelligence.vision.datasets.hf_split._support import fixture
from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf_split import (
    create_or_load_cardd_hf_split,
)


def test_corrupt_existing_manifest_is_never_silently_regenerated(tmp_path):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    path.write_text("partial JSON")
    with pytest.raises(DatasetValidationError, match="will not be regenerated"):
        create_or_load_cardd_hf_split(dataset, path)
    assert path.read_text() == "partial JSON"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"ratios": {"train": 0.7, "val": 0.3}},
        {"ratios": {"train": 0.5, "val": 0.1, "test": 0.1}},
        {"ratios": {"train": 0.7, "val": 0.3, "test": 0}},
        {"ratios": {"train": float("nan"), "val": 0.15, "test": 0.15}},
    ],
)
def test_invalid_split_parameters_rejected(tmp_path, kwargs):
    with pytest.raises(DatasetValidationError):
        create_or_load_cardd_hf_split(fixture(tmp_path), tmp_path / "split.json", **kwargs)
