"""Local tiny dataset fixtures; no official dataset or framework downloads."""

import json

import pytest
from PIL import Image

from vision.carparts_dataset import (
    MANIFEST_NAME,
    inspect_carparts_dataset,
    load_carparts_dataset,
)
from vision.dataset import DatasetValidationError


@pytest.fixture
def carparts_fixture(tmp_path):
    names = {0: "front_bumper", 1: "hood", 2: "object"}
    for split in ("train", "val", "test"):
        image = tmp_path / f"images/{split}/tiny.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (20, 15), "navy").save(image)
        label = tmp_path / f"labels/{split}/tiny.txt"
        label.parent.mkdir(parents=True)
        label.write_text("1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
    yaml = tmp_path / "dataset.yaml"
    yaml.write_text(
        json.dumps(
            {
                "path": str(tmp_path),
                "names": names,
                "nc": 3,
                **{split: f"images/{split}" for split in ("train", "val", "test")},
            }
        ),
        encoding="utf-8",
    )
    return yaml


def freeze_fixture(yaml):
    report = inspect_carparts_dataset(yaml)
    manifest = report | {"schema_version": 1}
    (yaml.parent / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_actual_yaml_classes_distribution_and_official_splits(carparts_fixture):
    report = freeze_fixture(carparts_fixture)
    assert report["valid"] and report["actual_sample_count"] == 3
    assert [row["name"] for row in report["category_mapping"]] == ["front_bumper", "hood", "object"]
    assert report["splits"]["test"]["image_count"] == 1
    assert report["splits"]["test"]["annotations_per_category"] == {"front_bumper": 0, "hood": 1, "object": 0}
    assert load_carparts_dataset(carparts_fixture) == report


@pytest.mark.parametrize("change", ["missing", "invalid", "bbox", "nan", "orphan", "empty", "broken_image"])
def test_inspection_reports_real_invalid_or_empty_labels(carparts_fixture, change):
    root = carparts_fixture.parent
    label = root / "labels/test/tiny.txt"
    if change == "missing":
        label.unlink()
    elif change == "orphan":
        (label.parent / "orphan.txt").write_text("", encoding="utf-8")
    elif change == "broken_image":
        (root / "images/test/tiny.png").write_bytes(b"broken image fixture")
    else:
        label.write_text(
            {"invalid": "99 0 0 1 0 1 1\n", "bbox": "0 0.5 0.5 0.3 0.3\n", "nan": "0 nan 0 1 0 1 1\n", "empty": ""}[
                change
            ],
            encoding="utf-8",
        )
    report = inspect_carparts_dataset(carparts_fixture)
    if change == "empty":
        assert report["valid"] and report["splits"]["test"]["images_without_annotation_count"] == 1
    else:
        assert not report["valid"]
    if change in {"invalid", "bbox", "nan"}:
        assert len(report["invalid_labels"]) == 1


def test_manifest_reinspection_refuses_changed_labels(carparts_fixture):
    freeze_fixture(carparts_fixture)
    (carparts_fixture.parent / "labels/test/tiny.txt").write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="changed since preparation"):
        load_carparts_dataset(carparts_fixture)


def test_missing_folder_is_not_success_and_never_downloads(tmp_path):
    yaml = tmp_path / "dataset.yaml"
    yaml.write_text(
        json.dumps(
            {
                "path": str(tmp_path),
                "names": ["hood"],
                **{split: f"images/{split}" for split in ("train", "val", "test")},
            }
        )
    )
    report = inspect_carparts_dataset(yaml)
    assert not report["valid"] and len(report["split_path_errors"]) == 3
    assert report["actual_sample_count"] == 0


def test_custom_split_path_is_rejected(carparts_fixture):
    document = json.loads(carparts_fixture.read_text())
    document["test"] = "images/train"
    carparts_fixture.write_text(json.dumps(document))
    with pytest.raises(DatasetValidationError, match="supplied"):
        inspect_carparts_dataset(carparts_fixture)
