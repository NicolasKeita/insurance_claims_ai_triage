"""Offline metadata tracking checks; no server, images or model weights."""

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vision.tracking import VisionMLflowTracking, vision_tracking_run
from vision.training import load_training_config


def tracker_stub():
    params, tags, metrics, artifacts = {}, {}, {}, []
    client = SimpleNamespace(
        log_params=params.update,
        set_tags=tags.update,
        set_tag=lambda key, value: tags.update({key: value}),
        log_metrics=metrics.update,
        log_artifact=lambda path, **kwargs: artifacts.append((path, kwargs)),
    )
    return VisionMLflowTracking(client, "offline-stub", log_checkpoint=False), params, tags, metrics, artifacts


def metric_fixture():
    return {
        "box": {"precision": 0.6, "recall": 0.4},
        "mask": {"map50_95": 0.3},
        "per_class": [
            {"name": "glass shatter", "damage_type": "glass_shatter", "box": {"precision": 0.5}, "mask": {"map50_95": 0.2}},
            {"name": "dent", "damage_type": "dent", "box": {"precision": None}, "mask": {"map50_95": None}},
        ],
    }


def test_training_tracking_preserves_frozen_provenance_and_validation_classes(tmp_path):
    tracker, params, tags, metrics, artifacts = tracker_stub()
    checkpoint = tmp_path / "train/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    results_csv = checkpoint.parent.parent / "results.csv"
    results_csv.write_text("epoch,loss\n1,0.1\n")
    summary_path = tmp_path / "training_summary.json"
    summary = {
        "config": {"seed": 42, "base_model": "yolo26n-seg.pt"},
        "environment": {"device_used": "cuda:0"},
        "dataset": {
            "source_name": "CarDD HF mirror", "source_repository": "harpreetsahota/CarDD",
            "source_revision": "stub-revision", "actual_sample_count": 2816,
            "split_version": 2, "official_splits_used": False,
            "split_manifest_sha256": "a" * 64, "prepared_manifest_sha256": "b" * 64,
            "dataset_yaml_sha256": "c" * 64, "annotation_fields": ["segmentations"],
            "category_mapping": [{"yolo_class_id": 0, "name": "glass shatter"}],
            "annotation_hashes": {"train": "d" * 64},
            "splits": {"train": {
                "image_count": 1971, "annotation_count": 4385,
                "annotations_per_category": {"glass shatter": 333},
                "images_per_category": {"glass shatter": 328},
            }},
            "metrics_scope": "offline fixture scope",
        },
        "best_checkpoint": "train/weights/best.pt", "best_checkpoint_sha256": "e" * 64,
        "confidence_threshold": 0.25, "best_epoch": 74,
        "epochs_completed": 94, "elapsed_seconds": 12.5,
        "validation_metrics": metric_fixture(),
    }
    tracker.log_training(summary, summary_path, checkpoint)
    assert params["dataset.source_repository"] == "harpreetsahota/CarDD"
    assert params["dataset.split_version"] == 2
    assert params["dataset.actual_sample_count"] == 2816
    assert params["dataset.train.sha256"] == "d" * 64
    assert params["dataset.train.glass shatter.source_images"] == 328
    assert params["training.best_epoch"] == 74
    assert tags["checkpoint_reference"] == "train/weights/best.pt"
    assert tags["registry_status"] == "deferred"
    assert metrics["validation.class.glass_shatter.mask.map50_95"] == 0.2
    assert "validation.class.dent.mask.map50_95" not in metrics
    assert metrics["training.epochs_completed"] == 94
    assert [Path(path).suffix for path, _ in artifacts] == [".json", ".csv"]


def test_test_tracking_logs_real_per_class_values_without_missing_class_imputation(tmp_path):
    tracker, _, tags, metrics, artifacts = tracker_stub()
    tracker.log_evaluation(metric_fixture(), [tmp_path / "test_metrics.json", tmp_path / "per_class_metrics.csv"])
    assert metrics["test.mask.map50_95"] == 0.3
    assert metrics["test.class.glass_shatter.mask.map50_95"] == 0.2
    assert "test.class.dent.mask.map50_95" not in metrics
    assert tags["final_evaluation_split"] == "test"
    assert len(artifacts) == 2


def test_optional_tracking_without_uri_does_not_block_cv(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config = replace(load_training_config(), track_mlflow=True)
    with pytest.warns(RuntimeWarning, match="not configured"):
        with vision_tracking_run(config) as tracker:
            assert tracker is None


def test_optional_server_failure_does_not_swallow_training_failure(monkeypatch):
    class StubMlflowException(Exception):
        pass

    def unavailable(_):
        raise StubMlflowException("offline stub server")

    fake = ModuleType("mlflow")
    fake.set_experiment = unavailable
    exceptions = ModuleType("mlflow.exceptions")
    exceptions.MlflowException = StubMlflowException
    configuration = ModuleType("ml.tracking")
    configuration.configure_mlflow_tracking = lambda: "http://offline.invalid"
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "ml.tracking", configuration)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://offline.invalid")
    with pytest.warns(RuntimeWarning, match="initialization failed"):
        with pytest.raises(RuntimeError, match="actual training failure"):
            with vision_tracking_run(load_training_config()) as tracker:
                assert tracker is None
                raise RuntimeError("actual training failure")
