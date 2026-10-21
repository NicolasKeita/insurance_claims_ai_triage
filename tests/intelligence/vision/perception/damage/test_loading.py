"""Test loading."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.intelligence.vision.perception.damage._constants import NAMES
from tests.intelligence.vision.perception.damage._support import framework_result
from vision.config import VisionSettings
from vision.damage.detector import (
    DamageDetectorError,
    UltralyticsDamageDetector,
)
from vision.images import ImageValidationError


def test_lazy_detector_loads_once_and_passes_explicit_device(tmp_path):
    checkpoint = tmp_path / "synthetic-test-checkpoint.pt"
    checkpoint.write_bytes(b"fake factory test only; not real weights")
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (20, 12), "white").save(path)
    calls, predictions = [], []

    def predict(**kwargs):
        predictions.append((kwargs["source"].size, kwargs["conf"], kwargs["device"]))
        return [framework_result()]

    def factory(checkpoint_path):
        calls.append(checkpoint_path)
        return SimpleNamespace(task="segment", names=NAMES, predict=predict)

    detector = UltralyticsDamageDetector(VisionSettings(model_path=checkpoint, device="cpu"), model_factory=factory)
    assert calls == []
    detector.assess_image(path)
    detector.assess_image(path)
    assert calls == [str(checkpoint.resolve())]
    assert predictions == [((20, 12), 0.25, "cpu")] * 2


def test_missing_checkpoint_and_invalid_image_never_invoke_model_factory(tmp_path):
    calls = []
    detector = UltralyticsDamageDetector(
        VisionSettings(model_path=tmp_path / "missing.pt"), model_factory=lambda path: calls.append(path)
    )
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (20, 12)).save(image)
    with pytest.raises(DamageDetectorError, match="No weights were downloaded"):
        detector.assess_image(image)
    image.write_bytes(b"corrupt")
    with pytest.raises(ImageValidationError):
        detector.assess_image(image)
    assert calls == []


@pytest.mark.parametrize("task,names", [("detect", NAMES), ("segment", {0: "car"})])
def test_detector_rejects_wrong_task_and_generic_pretrained_taxonomy(tmp_path, task, names):
    checkpoint = tmp_path / "synthetic-test-checkpoint.pt"
    checkpoint.write_bytes(b"stub test checkpoint")
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (20, 12)).save(path)
    detector = UltralyticsDamageDetector(
        VisionSettings(model_path=checkpoint),
        model_factory=lambda _: SimpleNamespace(task=task, names=names),
    )
    with pytest.raises(DamageDetectorError):
        detector.assess_image(path)


def test_domain_imports_do_not_import_ultralytics_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vision; import vision.damage.detector; import vision.service; "
            "assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
