"""Test adapters."""

import numpy as np
import pytest

from tests.vision.perception.bitmap._support import result
from vision.detector import (
    DamageMappingConfig,
    DetectorOutputError,
    map_ultralytics_result,
)
from vision.geometry import validated_polygon_geometry
from vision.models import VisionModelMetadata
from vision.parts_detector import (
    PartMappingConfig,
    map_ultralytics_part_result,
)


def test_both_adapters_prefer_actual_bitmap_and_preserve_its_hole():
    mask = np.ones((12, 20), dtype=bool)
    mask[3:8, 6:12] = False
    damage = map_ultralytics_result(
        result(mask, names={0: "scratch"}),
        config=DamageMappingConfig(
            filename="same.png",
            width=20,
            height=12,
            confidence_threshold=0.25,
            model_identifier="synthetic-damage",
        ),
    )
    parts = map_ultralytics_part_result(
        result(mask, names={0: "hood"}),
        config=PartMappingConfig(
            filename="same.png",
            width=20,
            height=12,
            confidence_threshold=0.25,
            model_metadata=VisionModelMetadata(identifier="synthetic-parts", framework="fake"),
        ),
    )
    for observation in (damage, parts):
        assert observation.detections[0].polygon.holes
        assert validated_polygon_geometry(observation.detections[0].polygon).area == pytest.approx(1 - 30 / 240)


def test_empty_mask_is_explicit_and_invalid_supplied_mask_is_not_boxed():
    empty = map_ultralytics_result(
        result(np.zeros((12, 20)), names={0: "scratch"}),
        config=DamageMappingConfig(
            filename="empty.png",
            width=20,
            height=12,
            confidence_threshold=0.25,
            model_identifier="fake",
        ),
    )
    assert empty.detections[0].polygon is None
    assert "empty binary mask" in empty.warnings[0]
    with pytest.raises(DetectorOutputError, match="binary"):
        map_ultralytics_result(
            result(np.full((12, 20), 0.4), names={0: "scratch"}),
            config=DamageMappingConfig(
                filename="bad.png",
                width=20,
                height=12,
                confidence_threshold=0.25,
                model_identifier="fake",
            ),
        )
