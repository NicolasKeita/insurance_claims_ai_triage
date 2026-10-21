"""Synthetic COCO fixtures test infrastructure, never real CarDD/model quality."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from vision.dataset import (
    DatasetValidationError,
    annotation_hash,
    build_dataset_manifest,
    discover_cardd_dataset,
    inspect_cardd_dataset,
    load_prepared_dataset,
    prepare_cardd_yolo,
    read_category_mapping,
    validate_segmentation_label,
)


def synthetic_cardd(tmp_path: Path) -> Path:
    """
    Tiny generated pixels with arbitrary polygons, not damage examples.

    Noncontiguous/reversed IDs deliberately avoid category_id - 1.
    """
    root = tmp_path / "external_coco"
    (root / "annotations").mkdir(parents=True)
    for index, split in enumerate(("train", "val", "test"), start=1):
        image_dir = root / f"{split}2017"
        image_dir.mkdir()
        filename = f"synthetic_{split}.png"
        Image.new("RGB", (20, 10), color=(index * 30, 0, 0)).save(image_dir / filename)
        document = {
            "images": [{"id": index, "file_name": filename, "width": 20, "height": 10}],
            "categories": [{"id": 42, "name": "scratch"}, {"id": 7, "name": "dent"}],
            "annotations": [
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 42,
                    "bbox": [2, 1, 10, 5],
                    "area": 50,
                    "iscrowd": 0,
                    "segmentation": [[2, 1, 12, 1, 12, 6, 2, 6]],
                }
            ],
        }
        (root / "annotations" / f"instances_{split}2017.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def edit_split(root: Path, split: str, change) -> None:
    path = root / "annotations" / f"instances_{split}2017.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def prepare_fixture(tmp_path: Path, **kwargs):
    root = synthetic_cardd(tmp_path)
    dataset = discover_cardd_dataset(root)
    output = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(
        dataset, output, manifest_path=tmp_path / "portable_manifest.json", converter="native", **kwargs
    )
    return root, output, manifest


def test_missing_environment_has_manual_download_instruction(monkeypatch):
    monkeypatch.delenv("CARDD_DATASET_DIR", raising=False)
    with pytest.raises(DatasetValidationError, match="set CARDD_DATASET_DIR"):
        discover_cardd_dataset()


def test_missing_directory_is_explicit(tmp_path):
    with pytest.raises(DatasetValidationError, match="does not exist"):
        discover_cardd_dataset(tmp_path / "missing")


def test_discovery_reads_real_categories_and_split_statistics(tmp_path, monkeypatch):
    root = synthetic_cardd(tmp_path)
    monkeypatch.setenv("CARDD_DATASET_DIR", str(root))
    summary = inspect_cardd_dataset(discover_cardd_dataset())
    assert summary["category_mapping"] == [
        {"category_id": 7, "name": "dent", "yolo_class_id": 0},
        {"category_id": 42, "name": "scratch", "yolo_class_id": 1},
    ]
    assert summary["splits"]["train"] == {
        "image_count": 1,
        "annotation_count": 1,
        "annotations_per_category": {"dent": 0, "scratch": 1},
        "images_per_category": {"dent": 0, "scratch": 1},
        "images_without_annotations": 0,
    }


@pytest.mark.parametrize("split,remove", [("train", "annotation"), ("val", "images"), ("test", "annotation")])
def test_requires_all_official_splits(tmp_path, split, remove):
    root = synthetic_cardd(tmp_path)
    if remove == "annotation":
        (root / "annotations" / f"instances_{split}2017.json").unlink()
    else:
        (root / f"{split}2017" / f"synthetic_{split}.png").unlink()
        (root / f"{split}2017").rmdir()
    with pytest.raises(DatasetValidationError, match=split):
        discover_cardd_dataset(root)


def test_unknown_image_reference_is_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(image_id=999))
    with pytest.raises(DatasetValidationError, match="unknown image ID"):
        discover_cardd_dataset(root)


def test_unknown_category_reference_is_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(category_id=999))
    with pytest.raises(DatasetValidationError, match="unknown category ID"):
        discover_cardd_dataset(root)


def test_annotation_requires_real_source_image(tmp_path):
    root = synthetic_cardd(tmp_path)
    (root / "train2017" / "synthetic_train.png").unlink()
    with pytest.raises(DatasetValidationError, match="Image file is missing"):
        discover_cardd_dataset(root)


def test_image_ids_cannot_overlap_across_splits(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "test", lambda doc: doc["images"][0].update(id=1))
    with pytest.raises(DatasetValidationError, match="Split leakage"):
        discover_cardd_dataset(root)


def test_filenames_cannot_overlap_across_splits(tmp_path):
    root = synthetic_cardd(tmp_path)
    (root / "test2017" / "synthetic_test.png").rename(root / "test2017" / "synthetic_train.png")
    edit_split(root, "test", lambda doc: doc["images"][0].update(file_name="synthetic_train.png"))
    with pytest.raises(DatasetValidationError, match="Split leakage"):
        discover_cardd_dataset(root)


def test_category_mapping_must_agree_across_splits(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "val", lambda doc: doc["categories"][0].update(name="crack"))
    with pytest.raises(DatasetValidationError, match="differ between splits"):
        discover_cardd_dataset(root)


@pytest.mark.parametrize(
    "categories",
    [
        [],
        [{"id": 2, "name": "dent"}, {"id": 2, "name": "scratch"}],
        [{"id": 2, "name": "dent"}, {"id": 3, "name": "DENT"}],
        [{"id": True, "name": "dent"}],
        [{"id": 2, "name": ""}],
    ],
)
def test_invalid_category_mapping_rejected(categories):
    with pytest.raises(DatasetValidationError):
        read_category_mapping({"categories": categories})


def test_no_annotation_images_are_measured_and_have_empty_labels(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "val", lambda doc: doc.update(annotations=[]))
    dataset = discover_cardd_dataset(root)
    assert inspect_cardd_dataset(dataset)["splits"]["val"]["images_without_annotations"] == 1
    prepared = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(dataset, prepared, manifest_path=None, converter="native")
    assert (prepared / "labels" / "val" / "synthetic_val.txt").read_text() == ""
    assert manifest["splits"]["val"]["converted_annotation_count"] == 0
    assert load_prepared_dataset(prepared / "dataset.yaml") == manifest


def test_manifest_hashes_are_portable_and_reproducible(tmp_path):
    root = synthetic_cardd(tmp_path)
    dataset = discover_cardd_dataset(root)
    first = build_dataset_manifest(dataset)
    second = build_dataset_manifest(dataset)
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["annotation_hashes"] == second["annotation_hashes"]
    assert (
        first["annotation_hashes"]["test"]
        == hashlib.sha256((root / "annotations" / "instances_test2017.json").read_bytes()).hexdigest()
    )
    assert "timestamp" in first and "preparation_version" in first
    assert str(root) not in json.dumps(first)
    assert not (root / "cardd_dataset_manifest.json").exists()
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(area=49))
    third = build_dataset_manifest(discover_cardd_dataset(root))
    assert third["dataset_fingerprint"] != first["dataset_fingerprint"]


def test_inspection_is_read_only(tmp_path):
    root = synthetic_cardd(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    inspect_cardd_dataset(discover_cardd_dataset(root))
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


def test_native_conversion_remaps_ids_and_preserves_polygon(tmp_path):
    root = synthetic_cardd(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(
        discover_cardd_dataset(root), output, manifest_path=tmp_path / "manifest.json", converter="native"
    )
    rows = validate_segmentation_label(
        output / "labels" / "train" / "synthetic_train.txt", class_count=2, expected_count=1
    )
    assert rows[0][0] == 1
    assert rows[0][1] == pytest.approx([0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6])
    assert load_prepared_dataset(output / "dataset.yaml") == manifest
    assert manifest["category_mapping"][1]["damage_type"] == "scratch"
    assert manifest["conversion"]["bbox_fallback"] is False
    assert str(root) not in json.dumps(manifest)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert json.loads((output / "dataset.yaml").read_text())["path"] == str(output.resolve())


def test_unsupported_domain_name_rejected_at_preparation_only(tmp_path):
    root = synthetic_cardd(tmp_path)
    for split in ("train", "val", "test"):
        edit_split(root, split, lambda doc: doc["categories"][0].update(name="unknown_physical_damage"))
    dataset = discover_cardd_dataset(root)
    assert inspect_cardd_dataset(dataset)["category_mapping"][1]["name"] == "unknown_physical_damage"
    with pytest.raises(DatasetValidationError, match="Unsupported domain taxonomy"):
        prepare_cardd_yolo(dataset, tmp_path / "prepared", manifest_path=None, converter="native")


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda ann: ann.pop("segmentation"), "missing_segmentation"),
        (lambda ann: ann.update(segmentation={"counts": "rle", "size": [10, 20]}), "rle"),
        (lambda ann: ann.update(iscrowd=1), "crowd"),
        (lambda ann: ann.update(segmentation=[[0, 0, 40, 1, 2, 2]]), "invalid_coordinates"),
        (lambda ann: ann.update(segmentation=[[0, 0, 1, 1, 2, 2]]), "degenerate_polygon"),
    ],
)
def test_unusable_masks_error_by_default_or_explicit_counted_skip(tmp_path, change, reason):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: change(doc["annotations"][0]))
    dataset = discover_cardd_dataset(root)
    output = tmp_path / "prepared"
    with pytest.raises(DatasetValidationError, match=reason):
        prepare_cardd_yolo(dataset, output, manifest_path=None, converter="native")
    assert not output.exists()
    manifest = prepare_cardd_yolo(dataset, output, manifest_path=None, converter="native", unsupported="skip")
    assert manifest["splits"]["train"]["skipped_annotations"] == {reason: 1}
    assert manifest["splits"]["train"]["skipped_annotations_per_category"] == {"scratch": 1}
    assert manifest["splits"]["train"]["converted_annotations_per_category"] == {"dent": 0, "scratch": 0}
    assert manifest["splits"]["train"]["converted_images_per_category"] == {"dent": 0, "scratch": 0}
    assert (output / "labels" / "train" / "synthetic_train.txt").read_text() == ""
    assert load_prepared_dataset(output / "dataset.yaml") == manifest


def test_multipart_preserves_component_vertices_and_records_limitation(tmp_path):
    root = synthetic_cardd(tmp_path)
    components = [[1, 1, 4, 1, 4, 4, 1, 4], [10, 1, 10, 4, 13, 4, 13, 1]]
    edit_split(root, "train", lambda doc: doc["annotations"][0].update(segmentation=components))
    output = tmp_path / "prepared"
    manifest = prepare_cardd_yolo(discover_cardd_dataset(root), output, manifest_path=None, converter="native")
    rows = validate_segmentation_label(output / "labels" / "train" / "synthetic_train.txt", class_count=2)
    polygon = rows[0][1]
    points = set(zip(polygon[::2], polygon[1::2]))
    assert len(rows) == 1
    assert all((part[i] / 20, part[i + 1] / 10) in points for part in components for i in range(0, len(part), 2))
    assert manifest["splits"]["train"]["stitched_multipart_annotations"] == 1
    assert "disconnected topology" in manifest["conversion"]["multipart_limitation"]


def test_invalid_multipart_component_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(
        root, "train", lambda doc: doc["annotations"][0].update(segmentation=[[1, 1, 4, 1, 4, 4], [10, 1, 10, 2]])
    )
    with pytest.raises(DatasetValidationError, match="invalid_polygon"):
        prepare_cardd_yolo(discover_cardd_dataset(root), tmp_path / "prepared", manifest_path=None, converter="native")


def test_duplicate_polygons_do_not_silently_disappear_in_yolo(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["annotations"].append(dict(doc["annotations"][0], id=123)))
    dataset = discover_cardd_dataset(root)
    with pytest.raises(DatasetValidationError, match="duplicate_polygon"):
        prepare_cardd_yolo(dataset, tmp_path / "prepared", manifest_path=None, converter="native")
    manifest = prepare_cardd_yolo(
        dataset, tmp_path / "prepared", manifest_path=None, converter="native", unsupported="skip"
    )
    assert manifest["splits"]["train"]["annotation_count"] == 2
    assert manifest["splits"]["train"]["converted_annotation_count"] == 1
    assert manifest["splits"]["train"]["skipped_annotations"] == {"duplicate_polygon": 1}


@pytest.mark.parametrize(
    "row", ["2 0 0 1 0 1 1\n", "0 0 0 1.2 0 1 1\n", "0 0 0 nan 0 1 1\n", "0 0 0 1 1\n", "0 0 0 0.5 0.5 1 1\n"]
)
def test_invalid_segmentation_rows_rejected(tmp_path, row):
    path = tmp_path / "label.txt"
    path.write_text(row)
    with pytest.raises(DatasetValidationError):
        validate_segmentation_label(path, class_count=2)


def test_prepared_label_tampering_is_rejected(tmp_path):
    _, output, _ = prepare_fixture(tmp_path)
    (output / "labels" / "train" / "synthetic_train.txt").write_text("0 0.1 0.1 0.6 0.1 0.6 0.6\n")
    with pytest.raises(DatasetValidationError, match="changed|disagree"):
        load_prepared_dataset(output / "dataset.yaml")


def test_prepared_unknown_or_orphan_files_rejected(tmp_path):
    _, output, _ = prepare_fixture(tmp_path)
    (output / "labels" / "test" / "orphan.txt").write_text("")
    with pytest.raises(DatasetValidationError, match="Unexpected or missing files"):
        load_prepared_dataset(output / "dataset.yaml")


def test_prepared_class_count_mismatch_rejected(tmp_path):
    _, output, _ = prepare_fixture(tmp_path)
    path = output / "dataset.yaml"
    document = json.loads(path.read_text())
    document["nc"] = 90
    path.write_text(json.dumps(document))
    with pytest.raises(DatasetValidationError, match="names/count"):
        load_prepared_dataset(path)


def test_annotation_image_dimensions_must_match_real_pixels(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["images"][0].update(width=21))
    with pytest.raises(DatasetValidationError, match="dimensions disagree"):
        prepare_cardd_yolo(discover_cardd_dataset(root), tmp_path / "prepared", manifest_path=None, converter="native")


def test_preparation_never_overwrites_or_writes_sources(tmp_path):
    root, output, _ = prepare_fixture(tmp_path)
    dataset = discover_cardd_dataset(root)
    with pytest.raises(DatasetValidationError, match="already exists"):
        prepare_cardd_yolo(dataset, output, manifest_path=None, converter="native")
    with pytest.raises(DatasetValidationError, match="outside and separate"):
        prepare_cardd_yolo(dataset, root / "prepared", manifest_path=None, converter="native")
    with pytest.raises(DatasetValidationError, match="manifest must be outside"):
        prepare_cardd_yolo(dataset, tmp_path / "another", manifest_path=root / "manifest.json", converter="native")


def test_source_path_traversal_is_rejected(tmp_path):
    root = synthetic_cardd(tmp_path)
    edit_split(root, "train", lambda doc: doc["images"][0].update(file_name="../synthetic_train.png"))
    with pytest.raises(DatasetValidationError, match="Unsafe relative filename"):
        discover_cardd_dataset(root)


def test_official_converter_is_lazy_and_auto_fallback_is_explicit(tmp_path, monkeypatch):
    import vision.dataset.preparation._converter as dataset_module

    root = synthetic_cardd(tmp_path)

    def unavailable(*args, **kwargs):
        raise ImportError("optional dependency unavailable")

    monkeypatch.setattr(dataset_module, "_try_ultralytics_conversion", unavailable)
    manifest = prepare_cardd_yolo(discover_cardd_dataset(root), tmp_path / "prepared", manifest_path=None)
    assert manifest["conversion"]["backend"] == "native_polygon"
    assert manifest["conversion"]["fallback_reason"].startswith("ImportError")


def test_module_import_does_not_import_ultralytics():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import vision.dataset; assert 'ultralytics' not in sys.modules"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_split_paths_support_custom_layout(tmp_path):
    root = synthetic_cardd(tmp_path)
    dataset = discover_cardd_dataset(
        root,
        annotation_files={split: f"annotations/instances_{split}2017.json" for split in ("train", "val", "test")},
        image_directories={split: f"{split}2017" for split in ("train", "val", "test")},
    )
    assert set(dataset.splits) == {"train", "val", "test"}


def test_annotation_hash_reads_bytes(tmp_path):
    path = tmp_path / "annotation.json"
    path.write_bytes(b"synthetic annotations\r\n")
    assert annotation_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()
