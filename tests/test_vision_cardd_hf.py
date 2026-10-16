"""Offline FiftyOne-shaped fixtures check mirror ingestion, not model quality."""

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vision.cardd_hf import (
    _area,
    _contains,
    _mask_polygons,
    _official_load,
    _PinnedHub,
    default_hf_cache_dir,
    inspect_cardd_hf,
    load_cardd_hf,
    normalize_fiftyone_dataset,
)
from vision.dataset import DatasetValidationError


class FixtureDataset:
    def __init__(self, samples, schema=None):
        self.samples = samples
        self.schema = schema or {
            "filepath": "StringField",
            "coco_id": "IntField",
            "boxes": "EmbeddedDocumentField(Detections)",
            "damage_regions": "EmbeddedDocumentField(Detections)",
        }

    def get_field_schema(self):
        return self.schema

    def iter_samples(self):
        return iter(self.samples)


def fixture(tmp_path: Path):
    image = tmp_path / "actual_image.png"
    Image.new("RGB", (20, 10), color="red").save(image)
    label = {
        "_cls": "Detection",
        "label": "glass shatter",
        "bounding_box": [0.1, 0.2, 0.5, 0.6],
        "mask": np.ones((3, 5), dtype=bool),
    }
    sample = {
        "filepath": str(image),
        "_id": "unstable-local-database-id",
        "coco_id": 7,
        "metadata": {"width": 20, "height": 10},
        "boxes": {"_cls": "Detections", "detections": [dict(label, mask=None)]},
        "damage_regions": {"_cls": "Detections", "detections": [label]},
    }
    return FixtureDataset([sample]), sample, label


def test_discovers_actual_instance_field_and_classes(tmp_path):
    raw, _, _ = fixture(tmp_path)
    dataset = normalize_fiftyone_dataset(raw, "owner/subset", "a" * 40)
    assert dataset.annotation_fields == ("damage_regions",)
    assert dataset.classes == ("glass shatter",)
    assert dataset.samples[0].sample_id == "coco:7"
    assert dataset.samples[0].image_sha256 == hashlib.sha256(dataset.samples[0].filepath.read_bytes()).hexdigest()
    summary = inspect_cardd_hf(dataset)
    assert summary["sample_count"] == 1
    assert summary["labeled_instance_count"] == 1
    assert summary["instances_per_class"] == {"glass shatter": 1}
    assert summary["segmentation_usable"] is True
    assert summary["official_cardd_splits_used"] is False
    assert summary["annotation_field_summaries"]["boxes"]["usable_for_instance_segmentation"] is False
    assert summary["annotation_field_summaries"]["boxes"]["source_labeled_instance_count"] == 1
    assert summary["annotation_field_summaries"]["damage_regions"]["per_instance_mask_count"] == 1
    assert summary["segmentation"]["instance_masks"] == 1
    assert summary["image_dimensions"]["width_min"] == 20


def test_source_identity_is_portable_and_ignores_local_fo_id(tmp_path):
    raw, sample, _ = fixture(tmp_path)
    del sample["coco_id"]
    first = normalize_fiftyone_dataset(raw, "owner/subset", None)
    sample["_id"] = "new-local-id"
    second = normalize_fiftyone_dataset(raw, "owner/subset", None)
    assert first.samples[0].sample_id == second.samples[0].sample_id
    assert first.samples[0].sample_id.startswith("image:")


def test_bbox_only_and_semantic_masks_fail_explicitly(tmp_path):
    raw, sample, label = fixture(tmp_path)
    del label["mask"]
    with pytest.raises(DatasetValidationError, match="lacks usable per-instance segmentation"):
        normalize_fiftyone_dataset(raw, "owner/subset", None)
    sample["semantic"] = {"_cls": "Segmentation", "mask": np.ones((10, 20), dtype=np.uint8)}
    raw.schema["semantic"] = "EmbeddedDocumentField(Segmentation)"
    with pytest.raises(DatasetValidationError, match="Semantic Segmentation alone"):
        normalize_fiftyone_dataset(raw, "owner/subset", None)


def test_actual_closed_filled_polyline_vertices_are_preserved(tmp_path):
    raw, sample, _ = fixture(tmp_path)
    raw.schema = {"boundaries": "EmbeddedDocumentField(Polylines)"}
    points = [[0.1, 0.2], [0.8, 0.2], [0.5, 0.7]]
    sample["boundaries"] = {
        "_cls": "Polylines",
        "polylines": [{"_cls": "Polyline", "label": "scratch", "closed": True, "filled": True, "points": [points]}],
    }
    dataset = normalize_fiftyone_dataset(raw, "owner/subset", None)
    assert dataset.samples[0].instances[0].polygons == (tuple(map(tuple, points)),)
    assert inspect_cardd_hf(dataset)["segmentation"]["polylines"] == 1


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda label: label.update(label="vehicle"), "Unsupported visual damage category"),
        (lambda label: label.update(mask=np.zeros((3, 5))), "no foreground"),
        (lambda label: label.update(mask=np.array([[0, 2], [1, 0]])), "binary pixels"),
        (lambda label: label.update(bounding_box=[0.9, 0.2, 0.5, 0.6]), "outside normalized"),
        (lambda label: label.update(mask=np.ones((3, 5, 1))), "2D binary"),
        (lambda label: label.update(iscrowd=1), "Crowd"),
    ],
)
def test_invalid_instance_geometry_fails_without_fake_masks(tmp_path, change, match):
    raw, _, label = fixture(tmp_path)
    change(label)
    with pytest.raises(DatasetValidationError, match=match):
        normalize_fiftyone_dataset(raw, "owner/subset", None)


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


def test_multiple_usable_fields_require_explicit_choice(tmp_path):
    raw, sample, label = fixture(tmp_path)
    sample["boxes"]["detections"][0] = label
    with pytest.raises(DatasetValidationError, match="Multiple usable annotation fields"):
        normalize_fiftyone_dataset(raw, "owner/subset", None)
    result = normalize_fiftyone_dataset(raw, "owner/subset", None, annotation_field="damage_regions")
    assert result.annotation_fields == ("damage_regions",)


@pytest.mark.parametrize(
    "what,match",
    [
        ("duplicate", "Duplicate stable sample ID"),
        ("dimension", "dimensions disagree"),
        ("missing", "Downloaded image is missing"),
    ],
)
def test_image_identity_and_actual_image_metadata_are_checked(tmp_path, what, match):
    raw, sample, _ = fixture(tmp_path)
    if what == "duplicate":
        raw.samples.append(sample)
    elif what == "dimension":
        sample["metadata"]["width"] = 10
    else:
        sample["filepath"] = str(tmp_path / "does_not_exist.png")
    with pytest.raises(DatasetValidationError, match=match):
        normalize_fiftyone_dataset(raw, "owner/subset", None)


def test_loader_uses_env_source_and_pinned_revision_offline(tmp_path, monkeypatch):
    raw, _, _ = fixture(tmp_path)
    monkeypatch.setenv("CARDD_HF_DATASET", "owner/subset")
    monkeypatch.setenv("CARDD_HF_REVISION", "tag")
    calls = []

    def resolve(repo, revision):
        calls.append(("resolve", repo, revision))
        return "b" * 40

    def load(repo, **kwargs):
        calls.append(("load", repo, kwargs))
        return raw

    result = load_cardd_hf(cache_dir=tmp_path / "cache", loader=load, revision_resolver=resolve)
    assert result.revision == "b" * 40
    assert calls == [("resolve", "owner/subset", "tag"), ("load", "owner/subset", {"revision": "b" * 40})]
    assert not (tmp_path / "cache").exists()


def test_pin_facade_forwards_sha_cache_and_rejects_wrong_revision(tmp_path):
    calls = []
    delegate = SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs), marker="original")
    proxy = _PinnedHub(delegate, "b" * 40, tmp_path)
    proxy.snapshot_download(repo_id="owner/subset", repo_type="dataset")
    assert calls == [
        {"repo_id": "owner/subset", "repo_type": "dataset", "revision": "b" * 40, "cache_dir": str(tmp_path)}
    ]
    assert proxy.marker == "original"
    with pytest.raises(DatasetValidationError, match="different HF revision"):
        proxy.snapshot_download(revision="c" * 40)


def test_failed_official_load_removes_only_its_new_db_entry_and_restores_config(tmp_path, monkeypatch):
    original_hub = SimpleNamespace(marker="original")
    fo = ModuleType("fiftyone")
    utils = ModuleType("fiftyone.utils")
    hub = ModuleType("fiftyone.utils.huggingface")
    entries, deleted = set(), []
    fo.config = SimpleNamespace(default_dataset_dir="unrelated_original_directory")
    fo.dataset_exists = lambda name: name in entries

    def delete(name):
        deleted.append(name)
        entries.remove(name)

    fo.delete_dataset = delete
    hub.hfh = original_hub

    def failing(repo, **kwargs):
        entries.add(kwargs["name"])
        assert isinstance(hub.hfh, _PinnedHub)
        assert fo.config.default_dataset_dir.startswith(str(tmp_path))
        raise RuntimeError("synthetic interrupted download")

    hub.load_from_hub = failing
    fo.utils, utils.huggingface = utils, hub
    for name, module in [("fiftyone", fo), ("fiftyone.utils", utils), ("fiftyone.utils.huggingface", hub)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("FIFTYONE_DATABASE_DIR", str(tmp_path / "db"))
    with pytest.raises(RuntimeError, match="interrupted download"):
        _official_load("owner/subset", "a" * 40, tmp_path)
    assert len(deleted) == 1 and deleted[0].startswith("cardd_hf_")
    assert not entries
    assert hub.hfh is original_hub
    assert fo.config.default_dataset_dir == "unrelated_original_directory"


def test_external_cache_setting_is_supported(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDD_HF_CACHE_DIR", str(tmp_path / "outside_checkout"))
    assert default_hf_cache_dir() == (tmp_path / "outside_checkout").resolve()


def test_fiftyone_is_not_imported_for_offline_fixtures():
    assert "fiftyone" not in sys.modules
