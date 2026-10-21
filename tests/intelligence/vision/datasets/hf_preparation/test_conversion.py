"""Test conversion."""

import json

import pytest

from tests.intelligence.vision.datasets.hf_preparation._support import prepare_fixture
from vision.dataset import (
    annotation_hash,
    load_prepared_dataset,
    validate_segmentation_label,
)
from vision.dataset.cardd_hf_preparation import (
    HF_PREPARATION_VERSION,
    prepare_cardd_hf_yolo,
)


def test_hf_preparation_preserves_all_instances_and_custom_split_provenance(tmp_path):
    source, output, manifest = prepare_fixture(tmp_path)
    assert manifest["preparation_version"] == HF_PREPARATION_VERSION
    assert manifest["source_repository"] == source.repository
    assert manifest["source_revision"] == source.revision
    assert manifest["actual_sample_count"] == 12
    assert manifest["annotation_fields"] == ["ground_truth"]
    assert manifest["field_schema"] == {"ground_truth": "Polylines"}
    assert manifest["official_splits_used"] is False
    assert manifest["conversion"]["bbox_fallback"] is False
    assert sum(split["converted_annotation_count"] for split in manifest["splits"].values()) == sum(
        len(sample.instances) for sample in source.samples
    )
    assert load_prepared_dataset(output / "dataset.yaml") == manifest
    by_id = {sample.sample_id: sample for sample in source.samples}
    for split, entries in manifest["prepared_files"].items():
        assert {entry["source_sample_id"] for entry in entries} == set(manifest["split_manifest"]["sample_ids"][split])
        for entry in entries:
            rows = validate_segmentation_label(output / entry["label"], class_count=2)
            sample = by_id[entry["source_sample_id"]]
            assert rows[0][0] == (0 if sample.instances[0].label == "dent" else 1)
            assert rows[0][1] == pytest.approx([0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6])
            assert annotation_hash(output / entry["image"]) == sample.image_sha256


def test_hf_preparation_reuses_frozen_split_and_portable_fingerprint(tmp_path):
    source, _, first = prepare_fixture(tmp_path)
    split_bytes = (tmp_path / "split_v1.json").read_bytes()
    second = prepare_cardd_hf_yolo(
        source,
        tmp_path / "prepared_hf_second",
        split_manifest_path=tmp_path / "split_v1.json",
        manifest_path=None,
        converter="native",
    )
    assert (tmp_path / "split_v1.json").read_bytes() == split_bytes
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["annotation_hashes"] == second["annotation_hashes"]
    assert str(tmp_path) not in json.dumps(first)
