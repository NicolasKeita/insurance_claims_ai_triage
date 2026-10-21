"""Test geometry."""

import numpy as np
import pytest
from shapely.geometry import Point

from vision.geometry import validated_polygon_geometry
from vision.masks import (
    BinaryMaskGeometryError,
    binary_mask_to_polygon,
    native_mask_polygons,
)


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
