from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from claims.case import ClaimCase
from claims.enums import DamageType
from claims.models import Claim
from vision.config import VisionSettings
from vision.detector import FakeDamageDetector
from vision.images import ImageValidationError, load_validated_image
from vision.models import (
    NO_DETECTION_SUMMARY,
    ImageDamageAssessment,
    NormalizedBoundingBox,
    NormalizedPoint,
    NormalizedPolygon,
    VisualDamageDetection,
)
from vision.service import ClaimVisionService
from vision.taxonomy import VisualDamageType, map_category_name
from vision.visualization import save_annotated_image


def detection(kind=VisualDamageType.DENT, confidence=0.8):
    return VisualDamageDetection(
        damage_type=kind,
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9),
        polygon=NormalizedPolygon(
            points=[NormalizedPoint(x=0.1, y=0.2), NormalizedPoint(x=0.8, y=0.2), NormalizedPoint(x=0.8, y=0.9)]
        ),
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("dent", VisualDamageType.DENT),
        ("Scratch", VisualDamageType.SCRATCH),
        ("crack", VisualDamageType.CRACK),
        ("glass shatter", VisualDamageType.GLASS_SHATTER),
        ("lamp_broken", VisualDamageType.LAMP_BROKEN),
        ("tire-flat", VisualDamageType.TIRE_FLAT),
    ],
)
def test_damage_mapping_uses_names_not_framework_ids(name, expected):
    assert map_category_name(name) == expected


@pytest.mark.parametrize("name", ["car", "bumper", "LEFT_HEADLIGHT", "shattered windshield", "", None])
def test_unknown_names_and_vehicle_parts_are_rejected(name):
    with pytest.raises(ValueError):
        map_category_name(name)
    assert VisualDamageType.LAMP_BROKEN.value != DamageType.LEFT_HEADLIGHT.value


@pytest.mark.parametrize(
    "changes",
    [
        {"x_min": -0.1},
        {"x_max": 1.1},
        {"y_min": float("nan")},
        {"x_min": 0.9, "x_max": 0.2},
        {"y_min": 0.5, "y_max": 0.5},
    ],
)
def test_invalid_boxes_fail(changes):
    fields = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}
    with pytest.raises(ValidationError):
        NormalizedBoundingBox(**(fields | changes))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_confidence_is_bounded(confidence):
    with pytest.raises(ValidationError):
        detection(confidence=confidence)


def test_polygon_rejects_degenerate_or_unbounded_points():
    with pytest.raises(ValidationError):
        NormalizedPolygon(points=[NormalizedPoint(x=0.1, y=0.1)] * 3)
    with pytest.raises(ValidationError):
        NormalizedPoint(x=1.1, y=0.2)


def test_zero_detection_summary_never_asserts_vehicle_is_undamaged():
    assessment = ImageDamageAssessment(filename="synthetic.png", width=8, height=8)
    assert assessment.detection_count == 0
    assert assessment.observation_summary == NO_DETECTION_SUMMARY
    assert assessment.model_dump()["detection_count"] == 0


def test_settings_environment_is_centralized(monkeypatch, tmp_path):
    monkeypatch.setenv("DAMAGE_MODEL_PATH", str(tmp_path / "trained.pt"))
    monkeypatch.setenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("DAMAGE_DEVICE", "cpu")
    settings = VisionSettings.from_env()
    assert settings.model_path == tmp_path / "trained.pt"
    assert settings.confidence_threshold == 0.4
    assert settings.device == "cpu"
    with pytest.raises(ValueError):
        VisionSettings(confidence_threshold=float("nan"))


def test_claim_service_keeps_per_image_observations_and_deduplicates_types(tmp_path):
    claim = Claim.model_validate_json(Path("data/CLAIM-2026-00001/claim.json").read_text(encoding="utf-8"))
    claim = claim.model_copy(update={"images": ["front.png", "left.png"]})
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for filename in ["front.png", "left.png", "undeclared.png"]:
        Image.new("RGB", (16, 12), "white").save(image_dir / filename)
    fake = FakeDamageDetector(
        {"front.png": [detection()], "left.png": [detection(), detection(VisualDamageType.SCRATCH)]}
    )
    result = ClaimVisionService(fake).assess_claim(ClaimCase(claim=claim, root_dir=tmp_path))
    assert [path.name for path in fake.calls] == ["front.png", "left.png"]
    assert [image.detection_count for image in result.images] == [1, 2]
    assert result.aggregate_damage_types == [VisualDamageType.DENT, VisualDamageType.SCRATCH]
    assert "damage_count" not in result.model_dump()
    assert result.model_metadata.framework == "fake"


def test_service_handles_empty_images_and_rejects_escaping_asset(tmp_path):
    claim = Claim.model_validate_json(Path("data/CLAIM-2026-00001/claim.json").read_text(encoding="utf-8"))
    service = ClaimVisionService(FakeDamageDetector())
    result = service.assess_claim(ClaimCase(claim=claim.model_copy(update={"images": []}), root_dir=tmp_path))
    assert result.images == [] and result.aggregate_damage_types == []
    with pytest.raises(ImageValidationError, match="outside"):
        service.assess_claim(
            ClaimCase(claim=claim.model_copy(update={"images": ["../outside.png"]}), root_dir=tmp_path)
        )


def test_image_exif_orientation_and_size_validation(tmp_path):
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (16, 10), "white").save(path, exif=exif)
    image = load_validated_image(path)
    assert image.size == (10, 16)
    image.close()
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(path, VisionSettings(max_image_pixels=100))
    with pytest.raises(ImageValidationError, match="bytes"):
        load_validated_image(path, VisionSettings(max_image_bytes=1))


@pytest.mark.parametrize("case", ["missing", "empty", "corrupt", "unsupported"])
def test_invalid_image_handling_without_model(tmp_path, case):
    path = tmp_path / ("image.gif" if case == "unsupported" else "image.png")
    if case == "empty":
        path.touch()
    elif case == "corrupt":
        path.write_bytes(b"this is not an image")
    elif case == "unsupported":
        Image.new("RGB", (8, 8)).save(path)
    fake = FakeDamageDetector()
    with pytest.raises(ImageValidationError):
        fake.assess_image(path)
    assert fake.calls == []


def test_debug_visualization_uses_domain_geometry_and_preserves_source(tmp_path):
    source = tmp_path / "synthetic.png"
    Image.new("RGB", (160, 120), "white").save(source)
    original = source.read_bytes()
    assessment = ImageDamageAssessment(filename=source.name, width=160, height=120, detections=[detection()])
    destination = save_annotated_image(source, assessment, tmp_path / "rendered/annotated.jpg")
    with Image.open(destination) as rendered:
        assert rendered.size == (160, 120)
        assert rendered.getpixel((120, 95)) != (255, 255, 255)
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="original"):
        save_annotated_image(source, assessment, source)
