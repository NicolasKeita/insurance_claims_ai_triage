"""Test resume."""

import json
from pathlib import Path

import pytest

from tests.intelligence.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.intelligence.vision.training.damage._dataset import prepared
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._resume import ResumeStub, interrupted_fixture
from vision.damage.training import (
    run_training,
)


def test_resume_retains_original_best_epoch_optimizer_stopper_and_total_budget(prepared):
    config, manifest = prepared
    metadata = interrupted_fixture(config, manifest)
    stub = ResumeStub(config, metadata)
    loaded = []

    def factory(path):
        loaded.append(path)
        return stub

    summary = run_training(
        config, model_factory=factory, environment_provider=lambda: {}, dataset_loader=lambda _: manifest, resume=True
    )
    last = str(config.output_dir / "train/weights/last.pt")
    assert loaded == [last] and stub.training_kwargs["resume"] == last
    assert stub.training_kwargs["save_dir"] == str(config.output_dir / "train")
    assert stub.training_kwargs["epochs"] == 100 and summary["epochs_completed"] == 100
    assert summary["best_epoch"] == 10 and summary["early_stopping_best_epoch"] == 8
    assert stub.trainer.best_fitness == stub.trainer.stopper.best_fitness == 1.0
    assert summary["base_checkpoint_sha256"] == metadata["base_checkpoint_sha256"]
    assert summary["best_checkpoint_sha256"] == metadata["best_checkpoint_sha256"]
    assert summary["elapsed_seconds_before_resume"] == 123.5 and summary["elapsed_seconds"] >= 123.5
    assert summary["resumed"] is True and summary["resume_from_epoch"] == 12
    assert not (config.output_dir / "train2").exists()


@pytest.mark.parametrize("change", ["config", "dataset", "threshold", "best", "last", "base", "completed", "evaluated"])
def test_resume_refuses_changed_or_completed_baseline_before_model_loading(prepared, change):
    config, manifest = prepared
    metadata = interrupted_fixture(config, manifest)
    if change == "config":
        metadata["config"]["seed"] = 999
    elif change == "dataset":
        metadata["dataset"]["annotation_hashes"]["val"] = "f" * 64
    elif change == "threshold":
        metadata["confidence_threshold"] = 0.5
    elif change in ("best", "last"):
        (config.output_dir / f"train/weights/{change}.pt").write_bytes(b"changed")
    elif change == "base":
        Path(metadata["base_checkpoint"]).write_bytes(b"changed")
    elif change == "completed":
        (config.output_dir / "training_summary.json").write_text("{}")
    else:
        (config.output_dir / "evaluation").mkdir()
    (config.output_dir / "recovery_training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume"):
        run_training(
            config,
            model_factory=lambda _: pytest.fail("must not load"),
            environment_provider=lambda: pytest.fail("must not import"),
            dataset_loader=lambda _: manifest,
            resume=True,
        )


def test_resume_rejects_stripped_checkpoint_without_optimizer(prepared):
    config, manifest = prepared
    stub = ResumeStub(config, interrupted_fixture(config, manifest))
    stub.ckpt["optimizer"] = None
    with pytest.raises(ValueError, match="optimizer state"):
        run_training(
            config,
            model_factory=lambda _: stub,
            environment_provider=lambda: {},
            dataset_loader=lambda _: manifest,
            resume=True,
        )
    assert stub.training_kwargs is None
