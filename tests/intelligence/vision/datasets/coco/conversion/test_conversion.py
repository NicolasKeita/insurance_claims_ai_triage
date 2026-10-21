"""Test conversion."""

import json

import pytest

from tests.intelligence.vision.datasets.coco._support import edit_split, synthetic_cardd
from vision.dataset import (
    DatasetValidationError,
    discover_cardd_dataset,
    inspect_cardd_dataset,
    load_prepared_dataset,
    prepare_cardd_yolo,
    validate_segmentation_label,
)


def test_native_conversion_remaps_ids_and_preserves_polygon(tmp_path):
    root = synthetic_cardd(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(
        discover_cardd_dataset(root), output, manifest_path=tmp_path / "manifest.json", converter="native"
    )
    rows = validate_segmentation_label(
        output / "labels" / "train" / "synthetic_train.txt", class_count=2, expected_count=1
    )
    assert rows[0][0] == 1
    assert rows[0][1] == pytest.approx([0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6])
    assert load_prepared_dataset(output / "dataset.yaml") == manifest
    assert manifest["category_mapping"][1]["damage_type"] == "scratch"
    assert manifest["conversion"]["bbox_fallback"] is False
    assert str(root) not in json.dumps(manifest)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert json.loads((output / "dataset.yaml").read_text())["path"] == str(output.resolve())


def test_unsupported_domain_name_rejected_at_preparation_only(tmp_path):
    root = synthetic_cardd(tmp_path)
    for split in ("train", "val", "test"):
        edit_split(root, split, lambda doc: doc["categories"][0].update(name="unknown_physical_damage"))
    dataset = discover_cardd_dataset(root)
    assert inspect_cardd_dataset(dataset)["category_mapping"][1]["name"] == "unknown_physical_damage"
    with pytest.raises(DatasetValidationError, match="Unsupported domain taxonomy"):
        prepare_cardd_yolo(dataset, tmp_path / "prepared", manifest_path=None, converter="native")


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda ann: ann.pop("segmentation"), "missing_segmentation"),
        (lambda ann: ann.update(segmentation={"counts": "rle", "size": [10, 20]}), "rle"),
        (lambda ann: ann.update(iscrowd=1), "crowd"),
        (lambda ann: ann.update(segmentation=[[0, 0, 40, 1, 2, 2]]), "invalid_coordinates"),
        (lambda ann: ann.update(segmentation=[[0, 0, 1, 1, 2, 2]]), "degenerate_polygon"),
    ],
)
def test_unusable_masks_error_by_default_or_explicit_counted_skip(tmp_path, change, reason):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: change(doc["annotations"][0]))
    dataset = discover_cardd_dataset(root)
    output = tmp_path / "prepared"
    with pytest.raises(DatasetValidationError, match=reason):
        prepare_cardd_yolo(dataset, output, manifest_path=None, converter="native")
    assert not output.exists()
    manifest = prepare_cardd_yolo(dataset, output, manifest_path=None, converter="native", unsupported="skip")
    assert manifest["splits"]["train"]["skipped_annotations"] == {reason: 1}
    assert manifest["splits"]["train"]["skipped_annotations_per_category"] == {"scratch": 1}
    assert manifest["splits"]["train"]["converted_annotations_per_category"] == {"dent": 0, "scratch": 0}
    assert manifest["splits"]["train"]["converted_images_per_category"] == {"dent": 0, "scratch": 0}
    assert (output / "labels" / "train" / "synthetic_train.txt").read_text() == ""
    assert load_prepared_dataset(output / "dataset.yaml") == manifest
