"""Test provenance."""

from dataclasses import replace

import pytest

from tests.intelligence.vision.datasets.hf_split._support import Instance, fixture
from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf_split import (
    create_or_load_cardd_hf_split,
)


@pytest.mark.parametrize(
    "change",
    [
        lambda dataset: replace(dataset, revision="b" * 40),
        lambda dataset: replace(dataset, repository="other/mirror"),
        lambda dataset: replace(dataset, classes=tuple(reversed(dataset.classes))),
        lambda dataset: replace(dataset, samples=dataset.samples[:-1]),
        lambda dataset: replace(
            dataset,
            samples=(
                replace(
                    dataset.samples[0], instances=(Instance("dent", polygons=(((0.2, 0.2), (0.9, 0.2), (0.9, 0.9)),)),)
                ),
                *dataset.samples[1:],
            ),
        ),
        lambda dataset: replace(dataset, annotation_fields=("other_field",)),
        lambda dataset: replace(
            dataset, samples=(replace(dataset.samples[0], sample_id="other-id"), *dataset.samples[1:])
        ),
    ],
)
def test_frozen_manifest_rejects_changed_source_without_replacement(tmp_path, change):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    before = path.read_bytes()
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        create_or_load_cardd_hf_split(change(dataset), path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kwargs", [{"seed": 43}, {"ratios": {"train": 0.6, "val": 0.2, "test": 0.2}}])
def test_frozen_manifest_rejects_changed_split_parameters(tmp_path, kwargs):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        create_or_load_cardd_hf_split(dataset, path, **kwargs)
