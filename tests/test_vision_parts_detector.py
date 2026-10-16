import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from vision.detector import DetectorOutputError
from vision.images import ImageValidationError
from vision.models import VisionModelMetadata
from vision.parts_detector import (
    PartMappingConfig,
    UltralyticsVehiclePartDetector,
    VehiclePartDetectorError,
    VehiclePartSettings,
    map_ultralytics_part_result,
    part_mapping_from_model_names,
)
from vision.taxonomy import CARPARTS_CLASS_TO_PART, VehiclePartType

NAMES = dict(enumerate(CARPARTS_CLASS_TO_PART))
METADATA = VisionModelMetadata(identifier="synthetic-parts", framework="fake", version="test-only")


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


def test_adapter_reuses_degenerate_contour_warning_and_rejects_crossing_mask():
    assessment = convert(framework_result(contour=[[0.1, 0.1], [0.3, 0.3], [0.5, 0.1], [0.3, 0.3]]))
    assert assessment.detections[0].polygon is None
    assert "zero-area segmentation contour" in assessment.warnings[0]
    with pytest.raises(DetectorOutputError, match="self-crossing"):
        convert(framework_result(contour=[[0, 0], [1, 1], [0, 1], [1, 0]]))
    with pytest.raises(DetectorOutputError, match="invalid normalized"):
        convert(framework_result(contour=[[0, 0], [1.1, 0], [1, 1]]))


def test_missing_masks_explicit_and_confidence_filter_preserved():
    result = framework_result()
    result.masks = None
    assessment = convert(result)
    assert assessment.detection_count == 1 and assessment.detections[0].polygon is None
    assert "unavailable" in assessment.warnings[0]
    assert convert(framework_result(confidence=0.2)).detections == []


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


def test_imports_do_not_load_models_frameworks_or_datasets():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vision.parts_detector; import vision.multimodal_service; "
            "assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
