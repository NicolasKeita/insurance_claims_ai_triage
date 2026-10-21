"""Test duplicates."""

from dataclasses import replace

from tests.vision.datasets.hf_split._support import assigned, fixture
from vision.cardd_hf_split import (
    create_or_load_cardd_hf_split,
    validate_cardd_hf_split,
)


def test_byte_identical_images_are_kept_together_without_losing_samples(tmp_path):
    dataset = fixture(tmp_path)
    original = dataset.samples[0]
    duplicate = dataset.samples[1]
    duplicate.filepath.write_bytes(original.filepath.read_bytes())
    dataset = replace(
        dataset, samples=(original, replace(duplicate, image_sha256=original.image_sha256), *dataset.samples[2:])
    )
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert assigned(manifest, original.sample_id) == assigned(manifest, duplicate.sample_id)
    assert manifest["duplicate_grouping"]["duplicate_image_sha256_groups"] == 1
    assert manifest["duplicate_grouping"]["samples_in_duplicate_groups"] == 2
    assert sum(len(ids) for ids in manifest["sample_ids"].values()) == 100
    assert validate_cardd_hf_split(dataset, manifest) == manifest


def test_same_filepath_different_ids_are_kept_together(tmp_path):
    dataset = fixture(tmp_path)
    original = dataset.samples[0]
    dataset = replace(
        dataset,
        samples=(
            original,
            replace(dataset.samples[1], filepath=original.filepath, image_sha256=original.image_sha256),
            *dataset.samples[2:],
        ),
    )
    manifest = create_or_load_cardd_hf_split(dataset, tmp_path / "split.json")
    assert assigned(manifest, "0") == assigned(manifest, "1")
    assert manifest["duplicate_grouping"]["duplicate_filepath_groups"] == 1
