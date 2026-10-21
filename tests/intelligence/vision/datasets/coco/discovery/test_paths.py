"""Test paths."""

import hashlib

from tests.intelligence.vision.datasets.coco._support import synthetic_cardd
from vision.dataset import (
    annotation_hash,
    discover_cardd_dataset,
)


def test_explicit_split_paths_support_custom_layout(tmp_path):
    root = synthetic_cardd(tmp_path)
    dataset = discover_cardd_dataset(
        root,
        annotation_files={split: f"annotations/instances_{split}2017.json" for split in ("train", "val", "test")},
        image_directories={split: f"{split}2017" for split in ("train", "val", "test")},
    )
    assert set(dataset.splits) == {"train", "val", "test"}


def test_annotation_hash_reads_bytes(tmp_path):
    path = tmp_path / "annotation.json"
    path.write_bytes(b"synthetic annotations\r\n")
    assert annotation_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()
