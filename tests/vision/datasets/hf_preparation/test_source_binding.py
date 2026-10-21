"""Test source binding."""

import json
import shutil

import pytest

from tests.vision.datasets.hf_preparation._support import prepare_fixture
from vision.cardd_hf_preparation import (
    hf_prepared_fingerprint,
)
from vision.dataset import (
    DatasetValidationError,
    annotation_hash,
    load_prepared_dataset,
    validate_segmentation_label,
)


def test_hf_wrong_image_with_updated_hash_is_bound_to_frozen_source(tmp_path):
    source, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["train"][0]
    original = next(sample for sample in source.samples if sample.sample_id == entry["source_sample_id"])
    replacement = next(
        sample
        for sample in source.samples
        if sample.sample_id != original.sample_id
        and [instance.label for instance in sample.instances] == [instance.label for instance in original.instances]
    )
    shutil.copy2(replacement.filepath, output / entry["image"])
    entry["image_sha256"] = replacement.image_sha256
    mapping = [
        {key: item[key] for key in ("category_id", "name", "yolo_class_id")} for item in manifest["category_mapping"]
    ]
    manifest["dataset_fingerprint"] = hf_prepared_fingerprint(manifest, mapping)
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="disagrees with frozen source sample"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_label_geometry_and_updated_hash_cannot_retain_old_fingerprint(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["train"][0]
    label = output / entry["label"]
    lines = label.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    fields[1] = "0.15"
    lines[0] = " ".join(fields)
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_segmentation_label(label, class_count=2, expected_count=entry["row_count"])
    entry["label_sha256"] = annotation_hash(label)
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="fingerprint disagrees"):
        load_prepared_dataset(output / "dataset.yaml")


@pytest.mark.parametrize("key,value", [("width", 21), ("height", 11), ("row_count", 100), ("class_counts", {"0": 100})])
def test_hf_prepared_dimensions_and_instance_counts_are_bound_to_frozen_source(tmp_path, key, value):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest["prepared_files"]["test"][0][key] = value
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="frozen source sample"):
        load_prepared_dataset(output / "dataset.yaml")
