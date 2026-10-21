"""Test geometry."""

import pytest

from tests.vision.perception.damage._support import convert, framework_result
from vision.detector import (
    DetectorOutputError,
)


def test_thin_v_mask_contour_keeps_detection_and_box_with_explicit_warning():
    """
    OpenCV CHAIN_APPROX_SIMPLE produces this retraced four-vertex contour for a one-pixel V mask, despite
    the underlying binary mask having pixels.
    """
    contour = [[1 / 12, 1 / 12], [3 / 12, 3 / 12], [5 / 12, 1 / 12], [3 / 12, 3 / 12]]
    assessment = convert(framework_result(polygon=contour))
    assert assessment.detection_count == 1
    assert assessment.detections[0].polygon is None
    assert assessment.detections[0].bounding_box.x_min == 0.1
    assert "zero-area segmentation contour" in assessment.warnings[0]


@pytest.mark.parametrize("invalid_coordinate", [float("nan"), float("inf"), -0.1, 1.1])
@pytest.mark.parametrize("point_count", [2, 4])
def test_invalid_coordinates_do_not_degrade_as_zero_area_contours(invalid_coordinate, point_count):
    with pytest.raises(DetectorOutputError, match="invalid normalized damage geometry"):
        convert(
            framework_result(
                polygon=[
                    [0.1, 0.1],
                    [invalid_coordinate, 0.2],
                    [0.3, 0.3],
                    [invalid_coordinate, 0.2],
                ][:point_count]
            )
        )
