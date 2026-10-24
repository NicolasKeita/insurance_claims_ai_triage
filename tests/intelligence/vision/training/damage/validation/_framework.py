"""Offline matching statistics; no trained weights, images or CV imports."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.intelligence.vision.training.damage._framework import OrchestrationStub, example_metric
from vision.damage.validation import ValidationParams


def diagnostic_params(manifest, stub=None, **overrides):
    return ValidationParams(
        dataset_loader=lambda _: manifest,
        model_factory=lambda _: stub or pytest.fail("must not load the model"),
        environment_provider=lambda: {},
        **overrides,
    )


class ValidationStub(OrchestrationStub):
    def __init__(self):
        super().__init__()
        self.calls = []

    def val(self, **kwargs):
        self.calls.append(kwargs)
        validator = SimpleNamespace(
            device="cpu",
            iouv=np.array([0.5]),
            metrics=SimpleNamespace(stats={key: [] for key in ("tp", "tp_m", "conf", "pred_cls", "target_cls")}),
        )
        for callback in self.callbacks.get("on_val_start", []):
            callback(validator)
        if kwargs["name"] == "operating":
            self._operating_batches(validator)
        result = example_metric()
        result.speed = {"preprocess": 0.1, "inference": 0.5, "loss": 0.0, "postprocess": 0.2}
        result.save_dir = Path(kwargs["project"]) / kwargs["name"]
        result.save_dir.mkdir(parents=True)
        return result

    def _operating_batches(self, validator):
        for row in (
            {
                "pred_cls": [1, 1],
                "target_cls": [1, 1],
                "conf": [0.8, 0.3],
                "tp": [[True], [True]],
                "tp_m": [[True], [False]],
            },
            {"pred_cls": [0], "target_cls": [], "conf": [0.9], "tp": [[False]], "tp_m": [[False]]},
            {"pred_cls": [], "target_cls": [], "conf": [], "tp": np.zeros((0, 1)), "tp_m": np.zeros((0, 1))},
        ):
            for key, value in row.items():
                validator.metrics.stats[key].append(np.asarray(value))
            for callback in self.callbacks.get("on_val_batch_end", []):
                callback(validator)
        self._clear_before_val_end(validator)

    @staticmethod
    def _clear_before_val_end(validator):
        """Match the production framework's clearing of statistics before on_val_end."""
        for values in validator.metrics.stats.values():
            values.clear()
