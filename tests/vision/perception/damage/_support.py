"""Support."""

from types import SimpleNamespace

from tests.vision.perception.damage._constants import NAMES
from vision.detector import (
    DamageMappingConfig,
    map_ultralytics_result,
)


def framework_result(*, confidence=0.8, class_id=1, box=None, polygon=None):
    return SimpleNamespace(
        names=NAMES,
        orig_shape=(12, 20),
        boxes=SimpleNamespace(xyxyn=[box or [0.1, 0.2, 0.8, 0.9]], cls=[class_id], conf=[confidence]),
        masks=SimpleNamespace(xyn=[polygon or [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9]]]),
    )


def convert(result, threshold=0.25):
    return map_ultralytics_result(
        result,
        config=DamageMappingConfig(
            filename="synthetic.png",
            width=20,
            height=12,
            confidence_threshold=threshold,
            model_identifier="test-only",
        ),
    )
