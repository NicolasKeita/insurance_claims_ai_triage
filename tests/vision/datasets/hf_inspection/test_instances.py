"""Test instances."""

import numpy as np
import pytest

from tests.vision.datasets.hf_inspection._support import fixture
from vision.cardd_hf import (
    inspect_cardd_hf,
    normalize_fiftyone_dataset,
)
from vision.dataset import DatasetValidationError


def test_bbox_only_and_semantic_masks_fail_explicitly(tmp_path):
    raw, sample, label = fixture(tmp_path)
    del label["mask"]
    with pytest.raises(DatasetValidationError, match="lacks usable per-instance segmentation"):
        normalize_fiftyone_dataset(raw, "owner/subset", None)
    sample["semantic"] = {"_cls": "Segmentation", "mask": np.ones((10, 20), dtype=np.uint8)}
    raw.schema["semantic"] = "EmbeddedDocumentField(Segmentation)"
    with pytest.raises(DatasetValidationError, match="Semantic Segmentation alone"):
        normalize_fiftyone_dataset(raw, "owner/subset", None)


def test_actual_closed_filled_polyline_vertices_are_preserved(tmp_path):
    raw, sample, _ = fixture(tmp_path)
    raw.schema = {"boundaries": "EmbeddedDocumentField(Polylines)"}
    points = [[0.1, 0.2], [0.8, 0.2], [0.5, 0.7]]
    sample["boundaries"] = {
        "_cls": "Polylines",
        "polylines": [{"_cls": "Polyline", "label": "scratch", "closed": True, "filled": True, "points": [points]}],
    }
    dataset = normalize_fiftyone_dataset(raw, "owner/subset", None)
    assert dataset.samples[0].instances[0].polygons == (tuple(map(tuple, points)),)
    assert inspect_cardd_hf(dataset)["segmentation"]["polylines"] == 1


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda label: label.update(label="vehicle"), "Unsupported visual damage category"),
        (lambda label: label.update(mask=np.zeros((3, 5))), "no foreground"),
        (lambda label: label.update(mask=np.array([[0, 2], [1, 0]])), "binary pixels"),
        (lambda label: label.update(bounding_box=[0.9, 0.2, 0.5, 0.6]), "outside normalized"),
        (lambda label: label.update(mask=np.ones((3, 5, 1))), "2D binary"),
        (lambda label: label.update(iscrowd=1), "Crowd"),
    ],
)
def test_invalid_instance_geometry_fails_without_fake_masks(tmp_path, change, match):
    raw, _, label = fixture(tmp_path)
    change(label)
    with pytest.raises(DatasetValidationError, match=match):
        normalize_fiftyone_dataset(raw, "owner/subset", None)
