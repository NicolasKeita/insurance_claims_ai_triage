"""Test metrics."""

from types import SimpleNamespace

import pytest

from tests.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.vision.training.damage._dataset import prepared
from tests.vision.training.damage._dataset import prepared as prepared
from tests.vision.training.damage._framework import example_metric
from vision.evaluation import extract_segmentation_metrics
from vision.training import (
    verify_class_names,
)


def test_metric_parser_does_not_fill_missing_classes_with_global_map(prepared):
    _, manifest = prepared
    parsed = extract_segmentation_metrics(example_metric(), manifest["category_mapping"], manifest["splits"]["test"])
    assert parsed["per_class"][0]["box"]["map50_95"] is None
    assert not parsed["per_class"][0]["mask_metrics_available"]
    assert parsed["per_class"][1]["mask"]["map50_95"] == 0.4
    assert parsed["weakest_classes_by_mask_map50_95"] == [
        {"name": "scratch", "mask_map50_95": 0.4, "annotation_count": 2}
    ]
    assert parsed["unavailable_classes"] == ["dent"]


def test_missing_metric_groups_and_nonfinite_values_remain_null(prepared):
    _, manifest = prepared
    result = SimpleNamespace(box=SimpleNamespace(mp=float("nan"), mr=None))
    parsed = extract_segmentation_metrics(result, manifest["category_mapping"], {})
    assert set(parsed["box"].values()) == {None}
    assert set(parsed["mask"].values()) == {None}


def test_metric_counts_use_converted_labels_and_preserve_source_distribution(prepared):
    _, manifest = prepared
    counts = dict(
        manifest["splits"]["test"],
        converted_annotations_per_category={"dent": 0, "scratch": 1},
        converted_images_per_category={"dent": 0, "scratch": 1},
    )
    parsed = extract_segmentation_metrics(example_metric(), manifest["category_mapping"], counts)
    assert parsed["per_class"][1]["annotation_count"] == 1
    assert parsed["per_class"][1]["source_annotation_count"] == 2


def test_invalid_metric_value_is_not_saved_as_a_score(prepared):
    _, manifest = prepared
    with pytest.raises(ValueError, match="outside"):
        extract_segmentation_metrics(SimpleNamespace(box=SimpleNamespace(mp=1.4)), manifest["category_mapping"], {})


def test_class_names_order_must_match_checkpoint_taxonomy(prepared):
    _, manifest = prepared
    verify_class_names({0: "dent", 1: "scratch"}, manifest["category_mapping"])
    with pytest.raises(ValueError, match="class names/order"):
        verify_class_names({0: "scratch", 1: "dent"}, manifest["category_mapping"])
