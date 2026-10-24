"""Validate split isolation, metric semantics and preservation of existing runs."""

import json

import pytest

from tests.intelligence.vision.training.damage._dataset import offline_environment as offline_environment
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._framework import completed_fixture
from tests.intelligence.vision.training.damage.validation._framework import ValidationStub, diagnostic_params
from vision.damage.validation import prepare_validation, run_validation


def test_two_val_passes_separate_ap_and_fixed_threshold_counts(prepared, tmp_path):
    config, manifest = prepared
    completed_fixture(config, manifest)
    (config.output_dir / "test_metrics.json").write_text("preserve TEST")
    before = {path: path.read_bytes() for path in config.output_dir.rglob("*") if path.is_file()}
    stub = ValidationStub()
    output_dir = tmp_path / "diagnostics"
    params = diagnostic_params(manifest, stub, image_size=960, batch_size=4)
    report = run_validation(config, output_dir=output_dir, params=params)
    assert [call["split"] for call in stub.calls] == ["val", "val"]
    assert [call["conf"] for call in stub.calls] == [0.001, 0.25]
    assert [call["plots"] for call in stub.calls] == [True, False]
    assert all(call["imgsz"] == 960 for call in stub.calls)
    assert all(call["batch"] == 4 for call in stub.calls)
    assert report["batch_size"] == 4 and report["diagnostic_config"]["batch"] == config.batch
    assert report["ap_speed_ms_per_image"]["inference"] == 0.5
    assert report["split"] == "val" and report["image_size"] == 960
    assert report["mask"]["map50"] == 0.6
    assert report["operating_point"]["mask"] == {"tp": 1, "fp": 2, "fn": 1, "precision": 1 / 3, "recall": 0.5}
    scratch = report["operating_point"]["per_class"][1]
    assert scratch["mask"]["recall"] == 0.5 and scratch["box"]["recall"] == 1
    assert report["operating_point"]["per_class"][0]["mask"]["recall"] is None
    assert "maximum mean F1" in report["metric_semantics"]
    assert json.loads((output_dir / "val_metrics.json").read_text())["status"] == "completed"
    assert (output_dir / "per_class_metrics.csv").is_file()
    assert (output_dir / "operating_point_metrics.csv").is_file()
    assert {path: path.read_bytes() for path in config.output_dir.rglob("*") if path.is_file()} == before


def test_check_only_validates_existing_evaluated_checkpoint_without_writes(prepared, tmp_path):
    config, manifest = prepared
    completed_fixture(config, manifest)
    (config.output_dir / "evaluation").mkdir()
    output_dir = tmp_path / "diagnostics"
    request = prepare_validation(config, output_dir=output_dir, params=diagnostic_params(manifest))
    assert request.checkpoint == config.output_dir / "train/weights/best.pt"
    assert request.payload()["validation_passes"] == 2
    assert not output_dir.exists()


def test_checkpoint_mutation_is_refused_before_loading(prepared, tmp_path):
    config, manifest = prepared
    completed_fixture(config, manifest)
    (config.output_dir / "train/weights/best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        run_validation(config, output_dir=tmp_path / "diagnostics", params=diagnostic_params(manifest))


def test_taxonomy_mismatch_is_refused_before_output_creation(prepared, tmp_path):
    config, manifest = prepared
    completed_fixture(config, manifest)
    stub = ValidationStub()
    stub.names = {0: "scratch", 1: "dent"}
    with pytest.raises(ValueError, match="taxonomy"):
        run_validation(config, output_dir=tmp_path / "diagnostics", params=diagnostic_params(manifest, stub))
    assert not (tmp_path / "diagnostics").exists()


def test_operating_threshold_can_be_changed_on_val_without_changing_training(prepared, tmp_path):
    config, manifest = prepared
    completed_fixture(config, manifest)
    report = run_validation(
        config,
        output_dir=tmp_path / "diagnostics",
        params=diagnostic_params(manifest, ValidationStub(), operating_confidence=0.2),
    )
    assert report["operating_confidence"] == 0.2
    summary = json.loads((config.output_dir / "training_summary.json").read_text())
    assert summary["confidence_threshold"] == 0.25


def test_missing_val_split_is_refused(prepared, tmp_path):
    config, manifest = prepared
    manifest = {**manifest, "splits": {}}
    with pytest.raises(ValueError, match="no VAL split"):
        prepare_validation(config, output_dir=tmp_path / "diagnostics", params=diagnostic_params(manifest))


def test_unexpected_framework_output_is_not_reported_as_completed(prepared, tmp_path):
    config, manifest = prepared
    completed_fixture(config, manifest)
    stub = ValidationStub()
    original = stub.val

    def redirected(**kwargs):
        result = original(**kwargs)
        result.save_dir = tmp_path / "unexpected"
        return result

    stub.val = redirected
    with pytest.raises(RuntimeError, match="unexpected directory"):
        run_validation(config, output_dir=tmp_path / "diagnostics", params=diagnostic_params(manifest, stub))
    assert not (tmp_path / "diagnostics/val_metrics.json").exists()
