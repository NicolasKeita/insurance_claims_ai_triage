"""Test geometry."""

from dataclasses import replace

import pytest

from tests.intelligence.vision.datasets.hf_preparation._support import (
    mirror_fixture,
    prepare_fixture,
)
from vision.dataset import DatasetValidationError, validate_segmentation_label
from vision.dataset.cardd_hf import HfDamageInstance


def test_hf_multipart_keeps_every_provided_vertex(tmp_path):
    dataset = mirror_fixture(tmp_path)
    instance = HfDamageInstance(
        "scratch",
        (((0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.1, 0.4)), ((0.6, 0.1), (0.6, 0.4), (0.8, 0.4), (0.8, 0.1))),
        "ground_truth",
        "polyline",
    )
    changed = replace(dataset.samples[0], instances=(instance,))
    dataset = replace(dataset, samples=(changed,) + dataset.samples[1:])
    _, output, manifest = prepare_fixture(tmp_path, dataset)
    entry = next(
        entry
        for entries in manifest["prepared_files"].values()
        for entry in entries
        if entry["source_sample_id"] == changed.sample_id
    )
    rows = validate_segmentation_label(output / entry["label"], class_count=2)
    points = set(zip(rows[0][1][::2], rows[0][1][1::2]))
    assert len(rows) == 1
    assert all(point in points for component in instance.polygons for point in component)
    assert sum(split["stitched_multipart_annotations"] for split in manifest["splits"].values()) == 1
    assert "disconnected topology" in manifest["conversion"]["multipart_limitation"]


def test_hf_source_annotations_do_not_become_bbox_only_labels(tmp_path):
    dataset = mirror_fixture(tmp_path)
    instance = HfDamageInstance("scratch", (), "ground_truth", "detection")
    dataset = replace(dataset, samples=(replace(dataset.samples[0], instances=(instance,)),) + dataset.samples[1:])
    with pytest.raises(DatasetValidationError, match="segmentation|polygon|geometry"):
        prepare_fixture(tmp_path, dataset)
    assert not (tmp_path / "prepared_hf").exists()
