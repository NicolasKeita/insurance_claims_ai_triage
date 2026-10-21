"""Test multipart."""

import pytest

from tests.vision.datasets.coco._support import edit_split, synthetic_cardd
from vision.dataset import (
    DatasetValidationError,
    discover_cardd_dataset,
    prepare_cardd_yolo,
    validate_segmentation_label,
)


def test_multipart_preserves_component_vertices_and_records_limitation(tmp_path):
    root = synthetic_cardd(tmp_path)
    components = [[1, 1, 4, 1, 4, 4, 1, 4], [10, 1, 10, 4, 13, 4, 13, 1]]
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(segmentation=components))
    output = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(discover_cardd_dataset(root), output, manifest_path=None, converter="native")
    rows = validate_segmentation_label(output / "labels" / "train" / "synthetic_train.txt", class_count=2)
    polygon = rows[0][1]
    points = set(zip(polygon[::2], polygon[1::2]))
    assert len(rows) == 1
    assert all((part[i] / 20, part[i + 1] / 10) in points for part in components for i in range(0, len(part), 2))
    assert manifest["splits"]["train"]["stitched_multipart_annotations"] == 1
    assert "disconnected topology" in manifest["conversion"]["multipart_limitation"]


def test_invalid_multipart_component_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(
        root, "train", lambda doc: doc["annotations"][0].update(segmentation=[[1, 1, 4, 1, 4, 4], [10, 1, 10, 2]])
    )
    with pytest.raises(DatasetValidationError, match="invalid_polygon"):
        prepare_cardd_yolo(discover_cardd_dataset(root), tmp_path / "prepared", manifest_path=None, converter="native")


def test_duplicate_polygons_do_not_silently_disappear_in_yolo(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["annotations"].append(dict(doc["annotations"][0], id=123)))
    dataset = discover_cardd_dataset(root)
    with pytest.raises(DatasetValidationError, match="duplicate_polygon"):
        prepare_cardd_yolo(dataset, tmp_path / "prepared", manifest_path=None, converter="native")
    manifest = prepare_cardd_yolo(
        dataset, tmp_path / "prepared", manifest_path=None, converter="native", unsupported="skip"
    )
    assert manifest["splits"]["train"]["annotation_count"] == 2
    assert manifest["splits"]["train"]["converted_annotation_count"] == 1
    assert manifest["splits"]["train"]["skipped_annotations"] == {"duplicate_polygon": 1}
