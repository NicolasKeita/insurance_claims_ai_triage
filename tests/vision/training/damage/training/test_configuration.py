"""Test configuration."""

from dataclasses import replace

import pytest

from tests.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.vision.training.damage._dataset import prepared
from tests.vision.training.damage._dataset import prepared as prepared
from vision.training import (
    load_training_config,
    run_training,
)


def test_versioned_baseline_and_environment_base_override(monkeypatch):
    config = load_training_config()
    assert (config.epochs, config.imgsz, config.batch, config.patience, config.seed) == (100, 640, 8, 20, 42)
    assert config.device == "0" and config.workers == 0 and config.optimizer == "auto"
    monkeypatch.setenv("DAMAGE_BASE_MODEL", "yolo26s-seg.pt")
    assert load_training_config().base_model == "yolo26s-seg.pt"


@pytest.mark.parametrize("model", ["yolo26n-seg.yaml", "scratch.yml", "weights.onnx"])
def test_config_rejects_scratch_or_non_pt_models(prepared, model):
    with pytest.raises(ValueError, match="pretrained .pt"):
        replace(prepared[0], base_model=model)


def test_dataset_validation_precedes_any_library_or_model_loading(prepared):
    config, _ = prepared
    calls = []

    def reject(_):
        raise ValueError("invalid prepared dataset")

    with pytest.raises(ValueError, match="invalid prepared"):
        run_training(
            config,
            dataset_loader=reject,
            model_factory=lambda _: calls.append("model"),
            environment_provider=lambda: calls.append("environment"),
        )
    assert calls == [] and not config.output_dir.exists()
