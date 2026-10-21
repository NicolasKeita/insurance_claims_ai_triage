"""Test runtime."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.intelligence.vision.training.vehicle_parts._support import PartStub
from vision.dataset.carparts_dataset import MANIFEST_NAME
from vision.parts.training import load_parts_training_config, run_parts_training


def test_cuda_available_but_framework_cpu_device_is_rejected(tmp_path):
    config = replace(
        load_parts_training_config(),
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "run",
        track_mlflow=False,
    )
    config.dataset_yaml.write_text("{}")
    (tmp_path / MANIFEST_NAME).write_text("{}")
    model = PartStub()

    def train(**_):
        for callback in model.callbacks["on_train_start"]:
            callback(SimpleNamespace(device="cpu", epoch=-1))
        pytest.fail("Runtime must refuse unexpected CPU before epochs")

    model.train = train
    with pytest.raises(RuntimeError, match="unexpected trainer device"):
        run_parts_training(
            config,
            model_factory=lambda _: model,
            environment_provider=lambda: {"cuda_available": True},
            dataset_loader=lambda _: {},
            tracking_factory=lambda *_args, **_kwargs: nullcontext(None),
        )
