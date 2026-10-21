"""Test masks."""

import numpy as np
import pytest

from vision.cardd_hf import (
    _area,
    _contains,
    _mask_polygons,
)


def test_mask_geometry_preserves_l_shape_and_exact_cell_area():
    mask = np.array([[1, 0, 0], [1, 1, 1]], dtype=bool)
    polygons, holes = _mask_polygons(mask, [0, 0, 1, 1])
    assert holes == 0
    assert len(polygons) == 1
    assert len(polygons[0]) == 6
    assert _area(polygons[0]) == pytest.approx(4 / 6)
    for row in range(2):
        for col in range(3):
            assert _contains(polygons[0], ((col + 0.5) / 3, (row + 0.5) / 2)) == bool(mask[row, col])


def test_mask_components_and_holes_preserve_foreground_area_and_centers():
    mask = np.ones((7, 9), dtype=bool)
    mask[2:4, 2:4] = False
    mask[:, 6] = False
    polygons, holes = _mask_polygons(mask, [0, 0, 1, 1])
    assert holes == 1
    assert len(polygons) == 2
    assert sum(_area(polygon) for polygon in polygons) == pytest.approx(mask.sum() / mask.size)
    for row in range(7):
        for col in range(9):
            assert any(_contains(polygon, ((col + 0.5) / 9, (row + 0.5) / 7)) for polygon in polygons) == bool(
                mask[row, col]
            )


def test_diagonally_touching_mask_pixels_use_eight_connected_foreground():
    polygons, holes = _mask_polygons(np.eye(2, dtype=bool), [0, 0, 1, 1])
    assert len(polygons) == 1
    assert holes == 0
    assert sum(_area(p) for p in polygons) == pytest.approx(0.5)


def test_all_small_binary_masks_preserve_area_and_pixel_membership():
    """
    Exhaustive small masks exercise diagonal contacts, holes and components.
    """
    for bits in range(1, 2**9):
        mask = np.array([(bits >> offset) & 1 for offset in range(9)], dtype=bool).reshape(3, 3)
        polygons, _ = _mask_polygons(mask, [0, 0, 1, 1])
        assert sum(_area(polygon) for polygon in polygons) == pytest.approx(mask.sum() / mask.size)
        for row in range(3):
            for col in range(3):
                assert any(_contains(polygon, ((col + 0.5) / 3, (row + 0.5) / 3)) for polygon in polygons) == bool(
                    mask[row, col]
                )
