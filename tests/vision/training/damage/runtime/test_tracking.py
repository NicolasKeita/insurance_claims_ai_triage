"""Test tracking."""

from types import SimpleNamespace

import pytest

from tests.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.vision.training.damage._dataset import prepared
from tests.vision.training.damage._dataset import prepared as prepared
from vision.tracking import VisionMLflowTracking, vision_tracking_run
from vision.training import (
    disable_builtin_mlflow_callbacks,
)


def test_builtin_image_logging_callbacks_removed_without_global_settings():
    def raw_image_logger(_):
        pytest.fail("raw images must not be logged")

    raw_image_logger.__module__ = "ultralytics.utils.callbacks.mlflow"

    def harmless(_):
        pass

    trainer = SimpleNamespace(
        callbacks={"on_pretrain_routine_end": [raw_image_logger, harmless], "on_train_end": [raw_image_logger]}
    )
    disable_builtin_mlflow_callbacks(trainer)
    assert trainer.callbacks == {"on_pretrain_routine_end": [harmless], "on_train_end": []}


def test_tracking_off_without_configuration_needs_no_mlflow(prepared):
    with vision_tracking_run(prepared[0]) as tracker:
        assert tracker is None


def test_tracking_rejects_image_artifacts(tmp_path):
    calls = []
    tracker = VisionMLflowTracking(
        SimpleNamespace(log_artifact=lambda *args, **kwargs: calls.append(args)), "stub", log_checkpoint=False
    )
    tracker.log_metadata_artifacts([tmp_path / "metrics.json"])
    with pytest.raises(ValueError, match="no CarDD images"):
        tracker.log_metadata_artifacts([tmp_path / "raw_car.jpg"])
    assert len(calls) == 1
