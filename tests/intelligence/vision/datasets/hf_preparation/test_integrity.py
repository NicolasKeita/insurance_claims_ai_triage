"""Test integrity."""

import json

import pytest
from PIL import Image

from tests.intelligence.vision.datasets.hf_preparation._support import prepare_fixture
from vision.dataset import (
    DatasetValidationError,
    load_prepared_dataset,
)


def test_hf_prepared_image_tampering_is_detected(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["test"][0]
    Image.new("RGB", (20, 10), color=(255, 255, 255)).save(output / entry["image"])
    with pytest.raises(DatasetValidationError, match="image content changed"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_prepared_labels_tampering_is_detected(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["train"][0]
    (output / entry["label"]).write_text("0 .1 .1 .5 .1 .5 .4\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="changed|disagree|count"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_frozen_split_tampering_is_detected(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest["split_manifest"]["sample_ids"]["test"].append("foreign-sample")
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="frozen split manifest hash"):
        load_prepared_dataset(output / "dataset.yaml")
