"""Test images."""

import pytest
from PIL import Image

from tests.intelligence.vision.perception.domain._support import detection
from vision.config import VisionSettings
from vision.damage.detector import FakeDamageDetector
from vision.images import ImageValidationError, load_validated_image
from vision.models import (
    ImageDamageAssessment,
)
from vision.visualization import save_annotated_image


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


def test_inference_thumbnail_bounds_native_masks_after_exif_without_changing_source(tmp_path):
    path = tmp_path / "large.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (1600, 800), "white").save(path, exif=exif)
    original = path.read_bytes()
    with load_validated_image(path, VisionSettings(inference_image_size=640)) as image:
        assert image.size == (320, 640)
    with load_validated_image(path) as image:
        assert image.size == (800, 1600)
    assert path.read_bytes() == original


def test_inference_thumbnail_does_not_upscale_small_images(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (20, 12)).save(path)
    with load_validated_image(path, VisionSettings(inference_image_size=640)) as image:
        assert image.size == (20, 12)


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
