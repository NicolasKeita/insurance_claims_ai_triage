"""Test loading."""

import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.intelligence.vision.perception.vehicle_parts._constants import NAMES
from tests.intelligence.vision.perception.vehicle_parts._support import framework_result
from vision.images import ImageValidationError
from vision.parts.detector import (
    UltralyticsVehiclePartDetector,
    VehiclePartDetectorError,
    VehiclePartSettings,
)


def test_part_detector_loads_local_checkpoint_once_and_versions_weights(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic factory input, not real weights")
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (20, 12)).save(image)
    loads, predictions = [], []

    def factory(path):
        loads.append(path)

        def predict(**options):
            predictions.append((options["source"].size, options["device"], options["conf"]))
            return [framework_result()]

        return SimpleNamespace(task="segment", names=NAMES, predict=predict)

    detector = UltralyticsVehiclePartDetector(
        VehiclePartSettings(model_path=checkpoint, device="cpu"), model_factory=factory
    )
    assert loads == []
    first, second = detector.assess_image(image), detector.assess_image(image)
    assert loads == [str(checkpoint.resolve())]
    assert predictions == [((20, 12), "cpu", 0.25)] * 2
    assert first.model_metadata == second.model_metadata
    assert first.model_metadata.version == f"sha256:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"


def test_part_model_missing_or_image_invalid_never_triggers_model_download(tmp_path):
    calls = []
    detector = UltralyticsVehiclePartDetector(
        VehiclePartSettings(model_path=tmp_path / "missing.pt"),
        model_factory=lambda path: calls.append(path),
    )
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (20, 12)).save(image)
    with pytest.raises(VehiclePartDetectorError, match="No weights were downloaded"):
        detector.assess_image(image)
    image.write_bytes(b"corrupt")
    with pytest.raises(ImageValidationError):
        detector.assess_image(image)
    assert calls == []
