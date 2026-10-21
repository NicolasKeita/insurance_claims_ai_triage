"""Test inventory."""

import pytest

from scripts.check_damage_test_smoke import select_test_entries


def test_smoke_inventory_covers_classes_deduplicates_and_bounds_selection():
    manifest = {
        "category_mapping": [{"yolo_class_id": index} for index in range(3)],
        "prepared_files": {
            "test": [
                {"image": "test/z.jpg", "class_counts": {"2": 1}},
                {"image": "test/a.jpg", "class_counts": {"0": 1, "1": 1}},
                {"image": "test/b.jpg", "class_counts": {"0": 1}},
                {"image": "test/d.jpg", "class_counts": {"1": 1}},
            ]
        },
    }
    selected = select_test_entries(
        manifest,
        maximum_images=3,
        diagnostic_examples=[{"filename": "test/a.jpg"}, {"filename": "test/d.jpg"}],
    )
    assert [entry["image"] for entry in selected] == ["test/a.jpg", "test/z.jpg", "test/d.jpg"]
    with pytest.raises(ValueError, match="absent from the frozen TEST"):
        select_test_entries(manifest, maximum_images=3, diagnostic_examples=[{"filename": "train/a.jpg"}])
    with pytest.raises(ValueError, match="positive"):
        select_test_entries(manifest, maximum_images=0)
