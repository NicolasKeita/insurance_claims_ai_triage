"""The native bitmap is evidence; lossy exterior contours cannot replace it."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point

from vision.config import VisionSettings
from vision.detector import DetectorOutputError, UltralyticsDamageDetector, map_ultralytics_result
from vision.geometry import validated_polygon_geometry
from vision.masks import BinaryMaskGeometryError, binary_mask_to_polygon, native_mask_polygons
from vision.models import VisionModelMetadata
from vision.parts_detector import map_ultralytics_part_result


def test_pixel_cells_preserve_thin_diagonal_holes_and_disconnected_regions():
    """
    diagonally touching single pixels
    """
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[1:8, 1:8] = 1
    mask[3:6, 3:6] = 0
    mask[9, 12] = mask[10, 13] = 1
    polygon = binary_mask_to_polygon(mask)
    geometry = validated_polygon_geometry(polygon)
    assert polygon.holes and len(polygon.components) == 2
    assert geometry.area == pytest.approx(mask.sum() / mask.size)
    for y, x in np.ndindex(mask.shape):
        assert geometry.contains(Point((x + 0.5) / 20, (y + 0.5) / 12)) == bool(mask[y, x])


@pytest.mark.parametrize("shape", [(3, 11), (11, 3)])
def test_native_non_square_mask_coordinates_are_normalized_to_the_same_image(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[-1, -1] = True
    polygon = native_mask_polygons(mask[None], count=1, width=shape[1], height=shape[0])[0]
    geometry = validated_polygon_geometry(polygon)
    assert geometry.bounds == pytest.approx(((shape[1] - 1) / shape[1], (shape[0] - 1) / shape[0], 1, 1))
    assert geometry.area == pytest.approx(1 / mask.size)


def test_padded_or_count_mismatched_masks_fail_instead_of_mixing_image_coordinates():
    with pytest.raises(BinaryMaskGeometryError, match="retina_masks"):
        native_mask_polygons(np.ones((1, 20, 20)), count=1, width=20, height=12)
    with pytest.raises(BinaryMaskGeometryError, match="shape"):
        native_mask_polygons(np.ones((2, 12, 20)), count=1, width=20, height=12)


@pytest.mark.parametrize("mask", [np.array([[np.nan]]), np.array([[0.4]]), np.array([[-1]]), np.zeros((0, 1))])
def test_invalid_binary_masks_are_rejected_without_bbox_substitution(mask):
    with pytest.raises(BinaryMaskGeometryError):
        binary_mask_to_polygon(mask)


class NativeMasks:
    def __init__(self, data):
        self.data = data

    @property
    def xyn(self):
        raise AssertionError("Native binary masks must not read lossy xyn contours")


def result(mask, *, names):
    return SimpleNamespace(
        names=names,
        orig_shape=(12, 20),
        masks=NativeMasks(mask[None]),
        boxes=SimpleNamespace(xyxyn=[[0.0, 0.0, 1.0, 1.0]], cls=[0], conf=[0.9]),
    )


def test_both_adapters_prefer_actual_bitmap_and_preserve_its_hole():
    mask = np.ones((12, 20), dtype=bool)
    mask[3:8, 6:12] = False
    damage = map_ultralytics_result(
        result(mask, names={0: "scratch"}),
        filename="same.png",
        width=20,
        height=12,
        confidence_threshold=0.25,
        model_identifier="synthetic-damage",
    )
    parts = map_ultralytics_part_result(
        result(mask, names={0: "hood"}),
        filename="same.png",
        width=20,
        height=12,
        confidence_threshold=0.25,
        model_metadata=VisionModelMetadata(identifier="synthetic-parts", framework="fake"),
    )
    for observation in (damage, parts):
        assert observation.detections[0].polygon.holes
        assert validated_polygon_geometry(observation.detections[0].polygon).area == pytest.approx(1 - 30 / 240)


def test_empty_mask_is_explicit_and_invalid_supplied_mask_is_not_boxed():
    empty = map_ultralytics_result(
        result(np.zeros((12, 20)), names={0: "scratch"}),
        filename="empty.png",
        width=20,
        height=12,
        confidence_threshold=0.25,
        model_identifier="fake",
    )
    assert empty.detections[0].polygon is None
    assert "empty binary mask" in empty.warnings[0]
    with pytest.raises(DetectorOutputError, match="binary"):
        map_ultralytics_result(
            result(np.full((12, 20), 0.4), names={0: "scratch"}),
            filename="bad.png",
            width=20,
            height=12,
            confidence_threshold=0.25,
            model_identifier="fake",
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
