"""Offline checks for mirror naming and provenance in the existing ML workflow."""

import json

from vision.training import dataset_provenance, load_training_config


def test_default_training_config_uses_hf_mirror_and_external_yaml_override(tmp_path, monkeypatch):
    path = tmp_path / "external" / "dataset.yaml"
    monkeypatch.setenv("DAMAGE_DATASET_YAML", str(path))
    config = load_training_config()
    assert config.version == "cardd_hf_yolo26n_seg_v1"
    assert config.dataset_yaml == path.resolve()
    assert config.epochs == 100 and config.seed == 42


def test_training_provenance_keeps_custom_split_scope_and_actual_source_fields(tmp_path):
    scope = (
        "Metrics are evaluated on the project's deterministic held-out split of the "
        "3-sample Hugging Face CarDD mirror, not on the official CarDD test split."
    )
    manifest = {
        "source_name": "CarDD HF mirror",
        "source_repository": "fixture/mirror",
        "source_revision": "a" * 40,
        "actual_sample_count": 3,
        "annotation_fields": ["inspected_instance_masks"],
        "official_splits_used": False,
        "split_version": "cardd_hf_split_v1",
        "split_manifest_sha256": "b" * 64,
        "metrics_scope": scope,
        "preparation_version": "cardd_hf_yolo_segmentation_v1",
        "annotation_hashes": {name: "c" * 64 for name in ("train", "val", "test")},
        "category_mapping": [{"category_id": 1, "name": "scratch", "yolo_class_id": 0}],
        "splits": {name: {"image_count": 1} for name in ("train", "val", "test")},
    }
    yaml = tmp_path / "dataset.yaml"
    yaml.write_text('{"path":"."}', encoding="utf-8")
    (tmp_path / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = dataset_provenance(yaml, manifest)
    for key in (
        "source_repository",
        "source_revision",
        "actual_sample_count",
        "annotation_fields",
        "official_splits_used",
        "split_version",
        "split_manifest_sha256",
        "metrics_scope",
    ):
        assert result[key] == manifest[key]
    assert len(result["prepared_manifest_sha256"]) == 64
