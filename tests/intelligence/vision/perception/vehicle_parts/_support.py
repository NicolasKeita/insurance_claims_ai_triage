"""Support."""

from types import SimpleNamespace

from tests.intelligence.vision.perception.vehicle_parts._constants import METADATA, NAMES
from vision.parts.detector import (
    PartMappingConfig,
    map_ultralytics_part_result,
)


def framework_result(*, name="hood", contour=None, confidence=0.9):
    return SimpleNamespace(
        names=NAMES,
        orig_shape=(12, 20),
        boxes=SimpleNamespace(xyxyn=[[0.1, 0.2, 0.8, 0.9]], cls=[list(NAMES.values()).index(name)], conf=[confidence]),
        masks=SimpleNamespace(xyn=[contour if contour is not None else [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9]]]),
    )


def convert(result):
    return map_ultralytics_part_result(
        result,
        config=PartMappingConfig(
            filename="synthetic.png",
            width=20,
            height=12,
            confidence_threshold=0.25,
            model_metadata=METADATA,
        ),
    )
