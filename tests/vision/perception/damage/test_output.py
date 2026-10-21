"""Test output."""

import pytest

from tests.vision.perception.damage._support import convert, framework_result
from vision.detector import (
    DetectorOutputError,
)
from vision.taxonomy import VisualDamageType


def test_framework_adapter_uses_names_and_compact_normalized_polygons():
    assessment = convert(framework_result(class_id=0))
    item = assessment.detections[0]
    assert item.damage_type == VisualDamageType.SCRATCH
    assert item.bounding_box.x_min == 0.1
    assert len(item.polygon.points) == 3
    assert item.model_identifier == "test-only"
    assert "mask" not in item.model_dump()


def test_threshold_filter_and_missing_mask_semantics():
    assert convert(framework_result(confidence=0.2)).detection_count == 0
    result = framework_result()
    result.masks = None
    assessment = convert(result)
    assert assessment.detections[0].polygon is None
    assert "unavailable" in assessment.warnings[0]


@pytest.mark.parametrize(
    "result",
    [
        framework_result(confidence=float("nan")),
        framework_result(confidence=1.1),
        framework_result(class_id=1.5),
        framework_result(class_id=99),
        framework_result(box=[-0.1, 0.2, 0.8, 0.9]),
        framework_result(polygon=[[0.1, 0.1], [float("nan"), 0.2], [0.3, 0.3]]),
        framework_result(polygon=[[0.1, 0.1], [1.1, 0.2], [0.3, 0.3]]),
    ],
)
def test_invalid_framework_output_is_rejected(result):
    with pytest.raises(DetectorOutputError):
        convert(result)
