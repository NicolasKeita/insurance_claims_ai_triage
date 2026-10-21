"""Dataset."""

import json

import pytest

from vision.training import (
    DamageTrainingConfig,
)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text('{"path":".","names":["dent","scratch"]}', encoding="utf-8")
    manifest = {
        "preparation_version": "test-fixture-only",
        "annotation_hashes": {split: str(index) * 64 for index, split in enumerate(("train", "val", "test"), 1)},
        "category_mapping": [
            {"category_id": 7, "yolo_class_id": 0, "name": "dent", "damage_type": "dent"},
            {"category_id": 42, "yolo_class_id": 1, "name": "scratch", "damage_type": "scratch"},
        ],
        "splits": {
            split: {
                "image_count": 3,
                "annotation_count": 2,
                "annotations_per_category": {"dent": 0, "scratch": 2},
                "images_per_category": {"dent": 0, "scratch": 1},
            }
            for split in ("train", "val", "test")
        },
    }
    (root / "cardd_dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = DamageTrainingConfig(
        version="fixture",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=100,
        seed=42,
        device="cpu",
        workers=0,
        optimizer="auto",
        base_model="yolo26n-seg.pt",
        dataset_yaml=dataset_yaml,
        output_dir=tmp_path / "run",
        experiment_name="fixture",
    )
    return config, manifest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DAMAGE_BASE_MODEL", raising=False)
    monkeypatch.delenv("DAMAGE_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("DAMAGE_DATASET_YAML", raising=False)
