"""Validation must report its runtime device, independent of loaded weights."""

from types import SimpleNamespace

import pytest

from vision.evaluation import EvaluationParams, run_evaluation
from vision.training import DamageTrainingConfig


@pytest.fixture
def evaluation_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.25")
    config = DamageTrainingConfig(
        version="device-fixture",
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        seed=42,
        device="0",
        workers=0,
        optimizer="auto",
        base_model="yolo26n-seg.pt",
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "run",
        experiment_name="device-fixture",
    )
    config.output_dir.mkdir()
    manifest = {
        "category_mapping": [{"category_id": 1, "yolo_class_id": 0, "name": "crack"}],
        "splits": {"test": {"image_count": 1, "annotation_count": 1}},
    }
    summary = {
        "confidence_threshold": 0.25,
        "best_checkpoint_sha256": "fixture-only",
        "dataset": manifest,
    }
    monkeypatch.setattr(
        "vision.evaluation._runtime._request.selected_checkpoint",
        lambda *_: (tmp_path / "fixture.pt", summary),
    )
    return config, manifest


class DeviceValidationStub:
    task = "segment"
    names = {0: "crack"}
    device = "cpu"

    def __init__(self, validator_device):
        self.validator_device = validator_device
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def val(self, **kwargs):
        """
        The caller never gets the validator, just its metric result.
        """
        validator = SimpleNamespace(device=self.validator_device)
        for callback in self.callbacks.get("on_val_start", []):
            callback(validator)
        return SimpleNamespace()


def evaluate_fixture(config, manifest, model):
    return run_evaluation(
        config,
        params=EvaluationParams(
            model_factory=lambda _: model,
            environment_provider=lambda: {"cuda_available": True},
            dataset_loader=lambda _: manifest,
            max_error_examples=0,
        ),
    )


@pytest.mark.parametrize("validator_device", ["cuda:0", "cpu"])
def test_reports_validator_device_instead_of_cpu_loaded_model(evaluation_fixture, validator_device):
    config, manifest = evaluation_fixture
    model = DeviceValidationStub(validator_device)
    result = evaluate_fixture(config, manifest, model)
    assert model.device == "cpu"
    assert result["evaluation_config"]["device"] == "0"
    assert result["environment"]["device_used"] == validator_device
    assert result["environment"]["device_used_source"] == "validator.device observed at on_val_start"


@pytest.mark.parametrize("callback_available", [True, False])
def test_unobserved_validator_device_never_infers_gpu_from_request(evaluation_fixture, callback_available):
    config, manifest = evaluation_fixture
    model = DeviceValidationStub(None)
    if not callback_available:
        model.add_callback = None
    result = evaluate_fixture(config, manifest, model)
    assert result["environment"]["device_used"] is None
    assert result["environment"]["loaded_checkpoint_device"] == "cpu"
    assert "unobserved" in result["environment"]["device_used_source"]
