"""Test service."""

from pathlib import Path

import pytest
from PIL import Image

from claims.case import ClaimCase
from claims.models import Claim
from tests.vision.perception.domain._support import detection
from vision.config import VisionSettings
from vision.detector import FakeDamageDetector
from vision.images import ImageValidationError
from vision.models import (
    NO_DETECTION_SUMMARY,
    ImageDamageAssessment,
)
from vision.service import ClaimVisionService
from vision.taxonomy import VisualDamageType


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
