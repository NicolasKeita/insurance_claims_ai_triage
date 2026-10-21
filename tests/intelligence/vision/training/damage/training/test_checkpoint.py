"""Test checkpoint."""

from types import SimpleNamespace

import pytest

from tests.intelligence.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.intelligence.vision.training.damage._dataset import prepared
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._framework import OrchestrationStub
from vision.damage.training import (
    run_training,
)


def test_best_epoch_tracks_actual_checkpoint_write_on_fitness_tie(prepared):
    config, manifest = prepared

    class TiedFitnessStub(OrchestrationStub):
        def train(self, **kwargs):
            result = super().train(**kwargs)
            self.trainer.epoch = 66
            self.trainer.fitness = self.trainer.best_fitness
            for callback in self.callbacks["on_model_save"]:
                callback(self.trainer)
            self.trainer.epoch = 99
            return result

    summary = run_training(
        config,
        model_factory=lambda _: TiedFitnessStub(),
        environment_provider=lambda: {},
        dataset_loader=lambda _: manifest,
    )
    assert summary["best_epoch"] == 67
    assert summary["early_stopping_best_epoch"] == 43


def test_existing_training_run_rejected_before_loading_model(prepared):
    config, manifest = prepared
    config.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_training(
            config,
            model_factory=lambda _: pytest.fail("must not load"),
            environment_provider=lambda: pytest.fail("must not import"),
            dataset_loader=lambda _: manifest,
        )


def test_base_checkpoint_task_must_be_segmentation(prepared):
    config, manifest = prepared
    with pytest.raises(ValueError, match="segmentation"):
        run_training(
            config,
            model_factory=lambda _: SimpleNamespace(task="detect"),
            environment_provider=lambda: {},
            dataset_loader=lambda _: manifest,
        )
    assert not config.output_dir.exists()
