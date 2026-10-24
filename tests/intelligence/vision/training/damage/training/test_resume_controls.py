"""An interrupted experiment must retain its explicit mask and augmentation controls."""

from dataclasses import replace

import pytest

from tests.intelligence.vision.training.damage._dataset import offline_environment as offline_environment
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._resume import ResumeStub, interrupted_fixture
from vision.damage.training import run_training


@pytest.mark.parametrize("control", ["mask_ratio", "mosaic", "scale", "close_mosaic"])
def test_resume_rejects_checkpoint_with_different_experiment_controls(prepared, control):
    config, manifest = prepared
    config = replace(config, mask_ratio=2, mosaic=0.5, scale=0.25, close_mosaic=20)
    model = ResumeStub(config, interrupted_fixture(config, manifest))
    model.ckpt["train_args"].update(config.training_overrides())
    model.ckpt["train_args"][control] = 999
    with pytest.raises(ValueError, match="training arguments differ"):
        run_training(
            config,
            model_factory=lambda _: model,
            environment_provider=lambda: {},
            dataset_loader=lambda _: manifest,
            resume=True,
        )
    assert model.training_kwargs is None
