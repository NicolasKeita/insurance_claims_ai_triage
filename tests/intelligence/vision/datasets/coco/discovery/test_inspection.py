"""Test inspection."""

import hashlib
import json

from tests.intelligence.vision.datasets.coco._support import edit_split, synthetic_cardd
from vision.dataset import (
    build_dataset_manifest,
    discover_cardd_dataset,
    inspect_cardd_dataset,
    load_prepared_dataset,
    prepare_cardd_yolo,
)


def test_no_annotation_images_are_measured_and_have_empty_labels(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "val", lambda doc: doc.update(annotations=[]))
    dataset = discover_cardd_dataset(root)
    assert inspect_cardd_dataset(dataset)["splits"]["val"]["images_without_annotations"] == 1
    prepared = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(dataset, prepared, manifest_path=None, converter="native")
    assert (prepared / "labels" / "val" / "synthetic_val.txt").read_text() == ""
    assert manifest["splits"]["val"]["converted_annotation_count"] == 0
    assert load_prepared_dataset(prepared / "dataset.yaml") == manifest


def test_manifest_hashes_are_portable_and_reproducible(tmp_path):
    root = synthetic_cardd(tmp_path)
    dataset = discover_cardd_dataset(root)
    first = build_dataset_manifest(dataset)
    second = build_dataset_manifest(dataset)
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["annotation_hashes"] == second["annotation_hashes"]
    assert (
        first["annotation_hashes"]["test"]
        == hashlib.sha256((root / "annotations" / "instances_test2017.json").read_bytes()).hexdigest()
    )
    assert "timestamp" in first and "preparation_version" in first
    assert str(root) not in json.dumps(first)
    assert not (root / "cardd_dataset_manifest.json").exists()
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(area=49))
    third = build_dataset_manifest(discover_cardd_dataset(root))
    assert third["dataset_fingerprint"] != first["dataset_fingerprint"]


def test_inspection_is_read_only(tmp_path):
    root = synthetic_cardd(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    inspect_cardd_dataset(discover_cardd_dataset(root))
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
