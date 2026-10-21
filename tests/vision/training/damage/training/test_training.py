"""Test training."""

import json

from tests.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.vision.training.damage._dataset import prepared
from tests.vision.training.damage._dataset import prepared as prepared
from tests.vision.training.damage._framework import completed_fixture
from vision.training import (
    file_sha256,
)


def test_training_stub_preserves_train_val_selection_and_exact_checkpoint(prepared):
    config, manifest = prepared
    stub, summary = completed_fixture(config, manifest)
    assert stub.training_kwargs["data"] == str(config.dataset_yaml)
    assert stub.training_kwargs["pretrained"] is True and stub.training_kwargs["val"] is True
    assert stub.training_kwargs["split"] == "val" and "test" not in stub.training_kwargs
    assert stub.training_kwargs["epochs"] == 100 and stub.training_kwargs["seed"] == 42
    assert stub.training_kwargs["plots"] is True
    assert summary["checkpoint_selection_split"] == "val"
    assert summary["best_checkpoint"] == "train/weights/best.pt"
    assert summary["best_epoch"] == 43
    assert summary["last_checkpoint"] == "train/weights/last.pt"
    assert summary["last_checkpoint_sha256"] != summary["best_checkpoint_sha256"]
    assert summary["test_used_during_training"] is False
    assert summary["early_stopping_triggered"] is False
    assert summary["epochs_completed"] == 100
    assert summary["environment"]["device_used"] == "cpu"
    assert summary["best_checkpoint_sha256"] == file_sha256(config.output_dir / summary["best_checkpoint"])
    assert len(list(config.output_dir.rglob("best.pt"))) == 1
    assert json.loads((config.output_dir / "training_progress.json").read_text())["status"] == "completed"
