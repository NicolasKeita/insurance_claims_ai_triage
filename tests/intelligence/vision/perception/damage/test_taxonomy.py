"""Test taxonomy."""

import pytest

from vision.damage.detector import (
    DetectorOutputError,
    category_mapping_from_model_names,
)


def test_mapping_rejects_generic_coco_unknown_classes_and_duplicates():
    with pytest.raises(DetectorOutputError, match="Unsupported"):
        category_mapping_from_model_names({0: "person", 1: "car"})
    with pytest.raises(DetectorOutputError, match="same damage"):
        category_mapping_from_model_names({0: "dent", 1: "dent"})
    with pytest.raises(DetectorOutputError, match="six"):
        category_mapping_from_model_names({0: "dent"}, require_complete=True)
