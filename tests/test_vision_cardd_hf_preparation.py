"""Synthetic mirror fixtures exercise geometry/provenance, never model quality."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from vision.cardd_hf import HfDamageDataset, HfDamageInstance, HfDamageSample
from vision.cardd_hf_preparation import HF_PREPARATION_VERSION, hf_prepared_fingerprint, prepare_cardd_hf_yolo
from vision.dataset import DatasetValidationError, annotation_hash, load_prepared_dataset, validate_segmentation_label


def mirror_fixture(tmp_path: Path) -> HfDamageDataset:
    source = tmp_path / "external_fiftyone_source"
    source.mkdir()
    samples = []
    for index in range(12):
        path = source / f"synthetic_{index}.png"
        Image.new("RGB", (20, 10), color=(index * 17, 40, 10)).save(path)
        polygons = (((0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)),)
        label = "dent" if index % 2 else "scratch"
        instances = (HfDamageInstance(label, polygons, "ground_truth", "polyline"),)
        if index % 3 == 0:
            other = "scratch" if label == "dent" else "dent"
            instances += (
                HfDamageInstance(
                    other, (((0.7, 0.1), (0.9, 0.1), (0.9, 0.4), (0.7, 0.4)),), "ground_truth", "polyline"
                ),
            )
        samples.append(HfDamageSample(f"stable-id-{index:02d}", path, 20, 10, instances, annotation_hash(path)))
    return HfDamageDataset(
        "synthetic/CarDD-mirror",
        "a" * 40,
        tuple(samples),
        {"ground_truth": "Polylines"},
        ("ground_truth",),
        ("dent", "scratch"),
    )


def prepare_fixture(tmp_path: Path, dataset=None):
    source = dataset or mirror_fixture(tmp_path)
    output = tmp_path / "prepared_hf"
    manifest = prepare_cardd_hf_yolo(
        source,
        output,
        split_manifest_path=tmp_path / "split_v1.json",
        manifest_path=tmp_path / "manifest.json",
        converter="native",
    )
    return source, output, manifest


def test_hf_preparation_preserves_all_instances_and_custom_split_provenance(tmp_path):
    source, output, manifest = prepare_fixture(tmp_path)
    assert manifest["preparation_version"] == HF_PREPARATION_VERSION
    assert manifest["source_repository"] == source.repository
    assert manifest["source_revision"] == source.revision
    assert manifest["actual_sample_count"] == 12
    assert manifest["annotation_fields"] == ["ground_truth"]
    assert manifest["field_schema"] == {"ground_truth": "Polylines"}
    assert manifest["official_splits_used"] is False
    assert manifest["conversion"]["bbox_fallback"] is False
    assert sum(split["converted_annotation_count"] for split in manifest["splits"].values()) == sum(
        len(sample.instances) for sample in source.samples
    )
    assert load_prepared_dataset(output / "dataset.yaml") == manifest
    by_id = {sample.sample_id: sample for sample in source.samples}
    for split, entries in manifest["prepared_files"].items():
        assert {entry["source_sample_id"] for entry in entries} == set(manifest["split_manifest"]["sample_ids"][split])
        for entry in entries:
            rows = validate_segmentation_label(output / entry["label"], class_count=2)
            sample = by_id[entry["source_sample_id"]]
            assert rows[0][0] == (0 if sample.instances[0].label == "dent" else 1)
            assert rows[0][1] == pytest.approx([0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6])
            assert annotation_hash(output / entry["image"]) == sample.image_sha256


def test_hf_preparation_reuses_frozen_split_and_portable_fingerprint(tmp_path):
    source, _, first = prepare_fixture(tmp_path)
    split_bytes = (tmp_path / "split_v1.json").read_bytes()
    second = prepare_cardd_hf_yolo(
        source,
        tmp_path / "prepared_hf_second",
        split_manifest_path=tmp_path / "split_v1.json",
        manifest_path=None,
        converter="native",
    )
    assert (tmp_path / "split_v1.json").read_bytes() == split_bytes
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["annotation_hashes"] == second["annotation_hashes"]
    assert str(tmp_path) not in json.dumps(first)


def test_hf_multipart_keeps_every_provided_vertex(tmp_path):
    dataset = mirror_fixture(tmp_path)
    instance = HfDamageInstance(
        "scratch",
        (((0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.1, 0.4)), ((0.6, 0.1), (0.6, 0.4), (0.8, 0.4), (0.8, 0.1))),
        "ground_truth",
        "polyline",
    )
    changed = replace(dataset.samples[0], instances=(instance,))
    dataset = replace(dataset, samples=(changed,) + dataset.samples[1:])
    _, output, manifest = prepare_fixture(tmp_path, dataset)
    entry = next(
        entry
        for entries in manifest["prepared_files"].values()
        for entry in entries
        if entry["source_sample_id"] == changed.sample_id
    )
    rows = validate_segmentation_label(output / entry["label"], class_count=2)
    points = set(zip(rows[0][1][::2], rows[0][1][1::2]))
    assert len(rows) == 1
    assert all(point in points for component in instance.polygons for point in component)
    assert sum(split["stitched_multipart_annotations"] for split in manifest["splits"].values()) == 1
    assert "disconnected topology" in manifest["conversion"]["multipart_limitation"]


def test_hf_source_annotations_do_not_become_bbox_only_labels(tmp_path):
    dataset = mirror_fixture(tmp_path)
    instance = HfDamageInstance("scratch", (), "ground_truth", "detection")
    dataset = replace(dataset, samples=(replace(dataset.samples[0], instances=(instance,)),) + dataset.samples[1:])
    with pytest.raises(DatasetValidationError, match="segmentation|polygon|geometry"):
        prepare_fixture(tmp_path, dataset)
    assert not (tmp_path / "prepared_hf").exists()


def test_hf_prepared_image_tampering_is_detected(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["test"][0]
    Image.new("RGB", (20, 10), color=(255, 255, 255)).save(output / entry["image"])
    with pytest.raises(DatasetValidationError, match="image content changed"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_prepared_labels_tampering_is_detected(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["train"][0]
    (output / entry["label"]).write_text("0 .1 .1 .5 .1 .5 .4\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="changed|disagree|count"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_frozen_split_tampering_is_detected(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest["split_manifest"]["sample_ids"]["test"].append("foreign-sample")
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="frozen split manifest hash"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_wrong_image_with_updated_hash_is_bound_to_frozen_source(tmp_path):
    source, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["train"][0]
    original = next(sample for sample in source.samples if sample.sample_id == entry["source_sample_id"])
    replacement = next(
        sample
        for sample in source.samples
        if sample.sample_id != original.sample_id
        and [instance.label for instance in sample.instances] == [instance.label for instance in original.instances]
    )
    shutil.copy2(replacement.filepath, output / entry["image"])
    entry["image_sha256"] = replacement.image_sha256
    mapping = [
        {key: item[key] for key in ("category_id", "name", "yolo_class_id")} for item in manifest["category_mapping"]
    ]
    manifest["dataset_fingerprint"] = hf_prepared_fingerprint(manifest, mapping)
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="disagrees with frozen source sample"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_label_geometry_and_updated_hash_cannot_retain_old_fingerprint(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    entry = manifest["prepared_files"]["train"][0]
    label = output / entry["label"]
    lines = label.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    fields[1] = "0.15"
    lines[0] = " ".join(fields)
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_segmentation_label(label, class_count=2, expected_count=entry["row_count"])
    entry["label_sha256"] = annotation_hash(label)
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="fingerprint disagrees"):
        load_prepared_dataset(output / "dataset.yaml")


@pytest.mark.parametrize("key,value", [("width", 21), ("height", 11), ("row_count", 100), ("class_counts", {"0": 100})])
def test_hf_prepared_dimensions_and_instance_counts_are_bound_to_frozen_source(tmp_path, key, value):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest["prepared_files"]["test"][0][key] = value
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="frozen source sample"):
        load_prepared_dataset(output / "dataset.yaml")


@pytest.mark.parametrize(
    "key,value", [("annotation_fields", ["fabricated_field"]), ("field_schema", {"ground_truth": "fabricated_schema"})]
)
def test_hf_prepared_schema_is_bound_to_frozen_source(tmp_path, key, value):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest[key] = value
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="fields/schema disagree"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_prepared_distribution_is_bound_to_frozen_source(tmp_path):
    _, output, manifest = prepare_fixture(tmp_path)
    manifest["splits"]["train"]["annotations_per_category"]["dent"] += 1
    (output / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="split distribution disagrees"):
        load_prepared_dataset(output / "dataset.yaml")


def test_hf_no_overwrite_and_no_source_modification(tmp_path):
    dataset = mirror_fixture(tmp_path)
    before = {sample.filepath: sample.filepath.read_bytes() for sample in dataset.samples}
    _, output, _ = prepare_fixture(tmp_path, dataset)
    assert {sample.filepath: sample.filepath.read_bytes() for sample in dataset.samples} == before
    with pytest.raises(DatasetValidationError, match="already exists"):
        prepare_cardd_hf_yolo(
            dataset, output, split_manifest_path=tmp_path / "split_v1.json", manifest_path=None, converter="native"
        )
    with pytest.raises(DatasetValidationError, match="outside and separate"):
        prepare_cardd_hf_yolo(
            dataset, dataset.samples[0].filepath.parent / "prepared", manifest_path=None, converter="native"
        )


def test_hf_preparation_module_does_not_import_frameworks_or_download():
    code = (
        "import sys; import vision.cardd_hf_preparation; "
        "assert not {'fiftyone', 'ultralytics', 'torch', 'huggingface_hub'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_hf_inspection_cli_requires_explicit_loading_mode():
    script = Path(__file__).resolve().parents[1] / "scripts" / "inspect_cardd_dataset.py"
    result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
    assert result.returncode == 2
    assert "--download" in result.stderr and "--dataset-name" in result.stderr
