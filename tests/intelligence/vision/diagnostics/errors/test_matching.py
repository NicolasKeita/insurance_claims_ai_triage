"""Test matching."""

import pytest

from vision.damage.error_analysis import (
    AnalysisConfig,
    ErrorInstance,
    analyze_instances,
    polygon_mask_iou,
)


def test_diagnostic_distinguishes_correct_box_with_bad_mask_from_good_mask():
    left = ((0.1, 0.1), (0.4, 0.1), (0.4, 0.9), (0.1, 0.9))
    right = ((0.6, 0.1), (0.9, 0.1), (0.9, 0.9), (0.6, 0.9))
    truth = [ErrorInstance(0, (0.1, 0.1, 0.9, 0.9), polygon=left)]
    bad = analyze_instances(
        [ErrorInstance(0, truth[0].box, 0.9, right)],
        truth,
        config=AnalysisConfig(confidence_threshold=0.25, image_size=(100, 100)),
    )
    assert bad["matches"][0]["box_iou"] == 1.0
    assert bad["segmentation_errors"][0]["polygon_mask_iou"] == 0.0
    assert bad["good_predictions"] == []
    assert bad["false_positives"] == bad["false_negatives"] == []
    good = analyze_instances(
        [ErrorInstance(0, truth[0].box, 0.9, left)],
        truth,
        config=AnalysisConfig(confidence_threshold=0.25, image_size=(100, 100)),
    )
    assert good["good_predictions"][0]["polygon_mask_iou"] == 1.0
    assert good["segmentation_errors"] == []


def test_missing_predicted_polygon_is_a_segmentation_failure():
    polygon = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9))
    result = analyze_instances(
        [ErrorInstance(0, (0.1, 0.1, 0.9, 0.9), 0.9)],
        [ErrorInstance(0, (0.1, 0.1, 0.9, 0.9), polygon=polygon)],
        config=AnalysisConfig(confidence_threshold=0.25, image_size=(20, 20)),
    )
    assert result["segmentation_errors"][0]["polygon_mask_iou"] == 0
    assert result["good_predictions"] == []
    with pytest.raises(ValueError, match="positive"):
        polygon_mask_iou(polygon, polygon, (0, 20))
