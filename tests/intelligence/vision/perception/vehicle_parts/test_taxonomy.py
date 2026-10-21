"""Test taxonomy."""

import pytest

from tests.intelligence.vision.perception.vehicle_parts._constants import METADATA
from tests.intelligence.vision.perception.vehicle_parts._support import convert, framework_result
from vision.damage.detector import DetectorOutputError
from vision.parts.detector import (
    part_mapping_from_model_names,
)
from vision.taxonomy import CARPARTS_CLASS_TO_PART, VehiclePartType


def test_part_mapping_uses_checkpoint_names_not_assumed_dataset_ids():
    mapping = part_mapping_from_model_names({0: "object", 1: "front_left_light", 2: "hood"})
    assert mapping == {0: VehiclePartType.OTHER, 1: VehiclePartType.FRONT_LEFT_LIGHT, 2: VehiclePartType.HOOD}
    detection = convert(framework_result(name="object")).detections[0]
    assert detection.part_type is VehiclePartType.OTHER
    assert detection.model_identifier == METADATA.identifier
    assert detection.model_version == METADATA.version


@pytest.mark.parametrize("names", [{0: "person"}, {0: "Hood"}, {0: "front light"}, {0: "hood", 1: "hood"}])
def test_generic_pretrained_changed_class_spellings_and_duplicate_classes_rejected(names):
    with pytest.raises(DetectorOutputError):
        part_mapping_from_model_names(names)


def test_checkpoint_requires_all_23_classes_but_not_hardcoded_ids():
    assert (
        len(
            part_mapping_from_model_names(
                dict(enumerate(reversed(list(CARPARTS_CLASS_TO_PART)))), require_complete=True
            )
        )
        == 23
    )
    with pytest.raises(DetectorOutputError, match="23"):
        part_mapping_from_model_names({0: "hood"}, require_complete=True)
