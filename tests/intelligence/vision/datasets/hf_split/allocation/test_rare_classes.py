"""Test rare classes."""

from dataclasses import replace

from tests.intelligence.vision.datasets.hf_split._support import Instance, fixture
from vision.dataset.cardd_hf_split import (
    create_or_load_cardd_hf_split,
)


def test_three_rare_class_images_represented_in_each_split(tmp_path):
    dataset = fixture(tmp_path)
    samples = tuple(
        replace(sample, instances=(Instance("dent"),) + ((Instance("crack"),) if int(sample.sample_id) < 3 else ()))
        for sample in dataset.samples
    )
    dataset = replace(dataset, samples=samples, classes=("dent", "crack"))
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert [manifest["splits"][name]["images_per_category"]["crack"] for name in ("train", "val", "test")] == [1, 1, 1]
    assert manifest["warnings"] == []


def test_impossible_rare_class_coverage_is_explicit(tmp_path):
    dataset = fixture(tmp_path)
    samples = tuple(
        replace(sample, instances=(Instance("dent"),) + ((Instance("crack"),) if sample.sample_id == "0" else ()))
        for sample in dataset.samples
    )
    manifest = create_or_load_cardd_hf_split(
        replace(dataset, samples=samples, classes=("dent", "crack")), tmp_path / "split.json"
    )
    assert any("'crack'" in warning and "impossible" in warning for warning in manifest["warnings"])
