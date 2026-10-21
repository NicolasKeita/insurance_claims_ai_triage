"""Test integration."""

import importlib.util
import sys

import pytest
from PIL import Image

from vision.config import VisionSettings
from vision.damage.detector import (
    UltralyticsDamageDetector,
)


@pytest.mark.integration
@pytest.mark.vision
def test_real_detector_with_existing_local_checkpoint(tmp_path):
    if importlib.util.find_spec("ultralytics") is None:
        pytest.fail(
            "Vision integration requires Ultralytics. Run it with "
            "python -m pytest -m vision from the configured .venv. "
            "No weights are downloaded."
        )
    settings = VisionSettings.from_env()
    if not settings.model_path.is_file():
        pytest.fail(
            f"Vision integration requires a trained local damage checkpoint: {settings.model_path}. "
            "Set DAMAGE_MODEL_PATH to existing local weights; no download is performed."
        )
    path = tmp_path / "synthetic-decoding-smoke.png"
    Image.new("RGB", (32, 24), "white").save(path)
    result = UltralyticsDamageDetector(settings).assess_image(path)
    assert (result.width, result.height) == (32, 24)
    assert result.detection_count == len(result.detections)
    print(f"Real vision inference completed with {sys.executable}")
