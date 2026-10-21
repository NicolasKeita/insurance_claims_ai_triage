"""Test allocation."""

import json
from dataclasses import replace

from tests.vision.datasets.hf_split._support import fixture
from vision.cardd_hf_split import (
    ALGORITHM,
    DEFAULT_SPLIT_MANIFEST_PATH,
    SPLIT_VERSION,
    create_or_load_cardd_hf_split,
    validate_cardd_hf_split,
)


def test_custom_multilabel_split_has_exact_counts_and_balanced_classes(tmp_path):
    dataset = fixture(tmp_path)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert manifest["target_counts"] == {"train": 70, "val": 15, "test": 15}
    assert {name: len(ids) for name, ids in manifest["sample_ids"].items()} == manifest["target_counts"]
    for name in ("train", "val", "test"):
        for label, total in (("dent", 100), ("scratch", 50), ("crack", 20)):
            assert abs(manifest["splits"][name]["images_per_category"][label] - total * manifest["ratios"][name]) <= 1
    assert manifest["warnings"] == []
    assert manifest["official_splits_used"] is False
    assert "HF mirror" in manifest["source_name"]
    assert set(manifest["annotation_hashes"]) == {"train", "val", "test"}
    assert validate_cardd_hf_split(dataset, manifest) == manifest
    assert manifest["split_version"] == SPLIT_VERSION == "cardd_hf_split_v2"
    assert manifest["algorithm"] == ALGORITHM
    assert DEFAULT_SPLIT_MANIFEST_PATH.name == "cardd_hf_split_v2.json"


def test_split_independent_of_input_order_and_cache_root(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = fixture(first_dir)
    second = fixture(second_dir)
    second = replace(second, samples=tuple(reversed(second.samples)))
    a = create_or_load_cardd_hf_split(first, tmp_path / "a.json")
    b = create_or_load_cardd_hf_split(second, tmp_path / "b.json")
    assert a == b
    assert str(tmp_path) not in json.dumps(a)
    assert validate_cardd_hf_split(second, a) == a
