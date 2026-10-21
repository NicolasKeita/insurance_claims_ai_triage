"""Test isolation."""

from dataclasses import replace

import pytest

from tests.intelligence.vision.datasets.hf_split._support import assigned, fixture
from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf_split import (
    create_or_load_cardd_hf_split,
    validate_cardd_hf_split,
)


def test_frozen_split_detects_identical_bytes_crossing_splits_after_id_swap(tmp_path):
    """
    IDs remain unique and split sizes unchanged; content still must not leak.
    """
    dataset = fixture(tmp_path)
    original, duplicate = dataset.samples[:2]
    duplicate.filepath.write_bytes(original.filepath.read_bytes())
    dataset = replace(
        dataset, samples=(original, replace(duplicate, image_sha256=original.image_sha256), *dataset.samples[2:])
    )
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    original_split = assigned(manifest, "0")
    other_split = next(name for name in ("train", "val", "test") if name != original_split)
    donor_id = manifest["sample_ids"][other_split][0]
    manifest["sample_ids"][original_split].remove("1")
    manifest["sample_ids"][original_split].append(donor_id)
    manifest["sample_ids"][other_split].remove(donor_id)
    manifest["sample_ids"][other_split].append("1")
    with pytest.raises(DatasetValidationError, match="identical image content"):
        validate_cardd_hf_split(dataset, manifest)


def test_duplicate_groups_incompatible_with_exact_sizes_fail_cleanly(tmp_path):
    dataset = fixture(tmp_path, count=6)
    samples = list(dataset.samples)
    for first, second in ((0, 1), (2, 3), (4, 5)):
        samples[second].filepath.write_bytes(samples[first].filepath.read_bytes())
        samples[second] = replace(samples[second], image_sha256=samples[first].image_sha256)
    path = tmp_path / "split.json"
    with pytest.raises(DatasetValidationError, match="cannot fit"):
        create_or_load_cardd_hf_split(replace(dataset, samples=tuple(samples)), path)
    assert not path.exists()


def test_duplicate_sample_ids_are_rejected(tmp_path):
    dataset = fixture(tmp_path)
    dataset = replace(
        dataset, samples=(dataset.samples[0], replace(dataset.samples[1], sample_id="0"), *dataset.samples[2:])
    )
    with pytest.raises(DatasetValidationError, match="duplicate sample ID"):
        create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")


def test_manifest_sample_overlap_and_distribution_tampering_rejected(tmp_path):
    dataset = fixture(tmp_path)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    manifest["sample_ids"]["val"][0] = manifest["sample_ids"]["train"][0]
    with pytest.raises(DatasetValidationError, match="leakage"):
        validate_cardd_hf_split(dataset, manifest)
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    manifest["splits"]["test"]["images_per_category"]["dent"] -= 1
    with pytest.raises(DatasetValidationError, match="distribution"):
        validate_cardd_hf_split(dataset, manifest)
