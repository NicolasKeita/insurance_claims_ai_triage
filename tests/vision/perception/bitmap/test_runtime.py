"""Test runtime."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from tests.vision.perception.bitmap._support import NativeMasks
from vision.config import VisionSettings
from vision.detector import (
    UltralyticsDamageDetector,
)
from vision.geometry import validated_polygon_geometry
from vision.masks import (
    binary_mask_to_polygon,
)


def test_detector_requests_official_native_scaling_after_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    with Image.new("RGB", (20, 12)) as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"local test factory input, not weights")

    def predict(**kwargs):
        assert kwargs["source"].size == (12, 20)
        assert kwargs["retina_masks"] is True
        return [
            SimpleNamespace(
                names={0: "scratch"},
                orig_shape=(20, 12),
                masks=NativeMasks(np.ones((1, 20, 12), dtype=bool)),
                boxes=SimpleNamespace(xyxyn=[[0.0, 0.0, 1.0, 1.0]], cls=[0], conf=[0.9]),
            )
        ]

    model = SimpleNamespace(
        task="segment",
        names=dict(enumerate(["scratch", "dent", "crack", "glass shatter", "lamp broken", "tire flat"])),
        predict=predict,
    )
    observation = UltralyticsDamageDetector(
        VisionSettings(model_path=checkpoint), model_factory=lambda _: model
    ).assess_image(path)
    assert (observation.width, observation.height) == (12, 20)
    assert validated_polygon_geometry(observation.detections[0].polygon).area == 1


def test_visualization_keeps_holes_transparent_and_secondary_components_visible():
    from vision.visualization import _paint_polygon

    mask = np.zeros((20, 20), dtype=bool)
    mask[1:13, 1:13] = True
    mask[4:10, 4:10] = False
    mask[16:19, 16:19] = True
    with Image.new("RGBA", (200, 200), (0, 0, 0, 0)) as image:
        _paint_polygon(image, binary_mask_to_polygon(mask), "red", 100)
        assert image.getpixel((60, 60))[3] == 0
        assert image.getpixel((20, 20))[3] == 100
        assert image.getpixel((170, 170))[3] == 100
