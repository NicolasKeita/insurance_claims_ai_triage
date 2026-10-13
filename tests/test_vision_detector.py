import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision.config import VisionSettings
from vision.detector import (
    DamageDetectorError, DetectorOutputError, UltralyticsDamageDetector,
    category_mapping_from_model_names, map_ultralytics_result,
)
from vision.images import ImageValidationError
from vision.taxonomy import VisualDamageType

NAMES = {0: "scratch", 1: "dent", 2: "crack", 3: "glass shatter", 4: "lamp broken", 5: "tire flat"}


def framework_result(*, confidence=0.8, class_id=1, box=None, polygon=None):
    return SimpleNamespace(
        names=NAMES, orig_shape=(12, 20),
        boxes=SimpleNamespace(xyxyn=[box or [0.1, 0.2, 0.8, 0.9]], cls=[class_id], conf=[confidence]),
        masks=SimpleNamespace(xyn=[polygon or [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9]]]),
    )


def convert(result, threshold=0.25):
    return map_ultralytics_result(result, filename="synthetic.png", width=20, height=12, confidence_threshold=threshold, model_identifier="test-only")


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


@pytest.mark.parametrize("result", [
    framework_result(confidence=float("nan")), framework_result(confidence=1.1),
    framework_result(class_id=1.5), framework_result(class_id=99),
    framework_result(box=[-0.1, 0.2, 0.8, 0.9]),
    framework_result(polygon=[[0.1, 0.1], [float("nan"), 0.2], [0.3, 0.3]]),
    framework_result(polygon=[[0.1, 0.1], [1.1, 0.2], [0.3, 0.3]]),
])
def test_invalid_framework_output_is_rejected(result):
    with pytest.raises(DetectorOutputError):
        convert(result)


def test_thin_v_mask_contour_keeps_detection_and_box_with_explicit_warning():
    # OpenCV CHAIN_APPROX_SIMPLE produces this retraced four-vertex contour for
    # a one-pixel V mask, despite the underlying binary mask having pixels.
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
        convert(framework_result(polygon=[
            [0.1, 0.1], [invalid_coordinate, 0.2], [0.3, 0.3], [invalid_coordinate, 0.2],
        ][:point_count]))


def test_mapping_rejects_generic_coco_unknown_classes_and_duplicates():
    with pytest.raises(DetectorOutputError, match="Unsupported"):
        category_mapping_from_model_names({0: "person", 1: "car"})
    with pytest.raises(DetectorOutputError, match="same damage"):
        category_mapping_from_model_names({0: "dent", 1: "dent"})
    with pytest.raises(DetectorOutputError, match="six"):
        category_mapping_from_model_names({0: "dent"}, require_complete=True)


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
    detector = UltralyticsDamageDetector(VisionSettings(model_path=tmp_path / "missing.pt"), model_factory=lambda path: calls.append(path))
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
        [sys.executable, "-c", "import sys; import vision; import vision.detector; import vision.service; assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.vision
def test_real_detector_opt_in_only_with_existing_local_checkpoint(tmp_path):
    if os.environ.get("RUN_VISION_TESTS") != "1":
        pytest.skip("Set RUN_VISION_TESTS=1 for local checkpoint integration; no download")
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("Ultralytics is not installed")
    settings = VisionSettings.from_env()
    if not settings.model_path.is_file():
        pytest.skip("Trained local damage checkpoint is unavailable")
    path = tmp_path / "synthetic-decoding-smoke.png"
    Image.new("RGB", (32, 24), "white").save(path)
    result = UltralyticsDamageDetector(settings).assess_image(path)
    assert (result.width, result.height) == (32, 24)
    assert result.detection_count == len(result.detections)
