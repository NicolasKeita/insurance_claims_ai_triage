"""Test output."""

import pytest

from tests.intelligence.vision.perception.vehicle_parts._support import convert, framework_result
from vision.damage.detector import DetectorOutputError


def test_adapter_reuses_degenerate_contour_warning_and_rejects_crossing_mask():
    assessment = convert(framework_result(contour=[[0.1, 0.1], [0.3, 0.3], [0.5, 0.1], [0.3, 0.3]]))
    assert assessment.detections[0].polygon is None
    assert "zero-area segmentation contour" in assessment.warnings[0]
    with pytest.raises(DetectorOutputError, match="self-crossing"):
        convert(framework_result(contour=[[0, 0], [1, 1], [0, 1], [1, 0]]))
    with pytest.raises(DetectorOutputError, match="invalid normalized"):
        convert(framework_result(contour=[[0, 0], [1.1, 0], [1, 1]]))


def test_missing_masks_explicit_and_confidence_filter_preserved():
    result = framework_result()
    result.masks = None
    assessment = convert(result)
    assert assessment.detection_count == 1 and assessment.detections[0].polygon is None
    assert "unavailable" in assessment.warnings[0]
    assert convert(framework_result(confidence=0.2)).detections == []
