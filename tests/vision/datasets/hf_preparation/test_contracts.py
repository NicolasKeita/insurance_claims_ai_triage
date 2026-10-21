"""Test contracts."""

import json

import pytest

from tests.vision.datasets.hf_preparation._support import prepare_fixture
from vision.dataset import (
    DatasetValidationError,
    load_prepared_dataset,
)


@pytest.mark.parametrize(
    "key,value", [("annotation_fields", ["fabricated_field"]), ("field_schema", {"ground_truth": "fabricated_schema"})]
)
def test_hf_prepared_schema_is_bound_to_frozen_source(tmp_path, key, value):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest[key] = value
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="fields/schema disagree"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_prepared_distribution_is_bound_to_frozen_source(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest["splits"]["train"]["annotations_per_category"]["dent"] += 1
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="split distribution disagrees"):
        load_prepared_dataset(output / "dataset.yaml")
