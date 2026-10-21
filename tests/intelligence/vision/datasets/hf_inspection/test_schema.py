"""Test schema."""

import hashlib

from tests.intelligence.vision.datasets.hf_inspection._support import fixture
from vision.dataset.cardd_hf import (
    inspect_cardd_hf,
    normalize_fiftyone_dataset,
)


def test_discovers_actual_instance_field_and_classes(tmp_path):
    raw, _, _ = fixture(tmp_path)
    dataset = normalize_fiftyone_dataset(raw, "owner/subset", "a" * 40)
    assert dataset.annotation_fields == ("damage_regions",)
    assert dataset.classes == ("glass shatter",)
    assert dataset.samples[0].sample_id == "coco:7"
    assert dataset.samples[0].image_sha256 == hashlib.sha256(dataset.samples[0].filepath.read_bytes()).hexdigest()
    summary = inspect_cardd_hf(dataset)
    assert summary["sample_count"] == 1
    assert summary["labeled_instance_count"] == 1
    assert summary["instances_per_class"] == {"glass shatter": 1}
    assert summary["segmentation_usable"] is True
    assert summary["official_cardd_splits_used"] is False
    assert summary["annotation_field_summaries"]["boxes"]["usable_for_instance_segmentation"] is False
    assert summary["annotation_field_summaries"]["boxes"]["source_labeled_instance_count"] == 1
    assert summary["annotation_field_summaries"]["damage_regions"]["per_instance_mask_count"] == 1
    assert summary["segmentation"]["instance_masks"] == 1
    assert summary["image_dimensions"]["width_min"] == 20


def test_source_identity_is_portable_and_ignores_local_fo_id(tmp_path):
    raw, sample, _ = fixture(tmp_path)
    del sample["coco_id"]
    first = normalize_fiftyone_dataset(raw, "owner/subset", None)
    sample["_id"] = "new-local-id"
    second = normalize_fiftyone_dataset(raw, "owner/subset", None)
    assert first.samples[0].sample_id == second.samples[0].sample_id
    assert first.samples[0].sample_id.startswith("image:")
