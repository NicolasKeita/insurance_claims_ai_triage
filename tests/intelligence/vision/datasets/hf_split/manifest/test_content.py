"""Test content."""

import hashlib
from dataclasses import replace

import pytest

from tests.intelligence.vision.datasets.hf_split._support import fixture
from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf_split import (
    create_or_load_cardd_hf_split,
)


def test_changed_actual_image_bytes_fail_even_if_declared_hash_is_stale(tmp_path):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    dataset.samples[0].filepath.write_bytes(b"changed pixels")
    with pytest.raises(DatasetValidationError, match="content changed"):
        create_or_load_cardd_hf_split(dataset, path)


def test_changed_content_with_updated_hash_rejects_frozen_manifest(tmp_path):
    dataset = fixture(tmp_path)
    path = tmp_path / "split.json"
    create_or_load_cardd_hf_split(dataset, path)
    dataset.samples[0].filepath.write_bytes(b"changed pixels")
    dataset = replace(
        dataset,
        samples=(
            replace(dataset.samples[0], image_sha256=hashlib.sha256(b"changed pixels").hexdigest()),
            *dataset.samples[1:],
        ),
    )
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        create_or_load_cardd_hf_split(dataset, path)
