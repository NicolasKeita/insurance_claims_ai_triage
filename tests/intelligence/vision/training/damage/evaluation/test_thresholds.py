"""Test thresholds."""

import pytest

from tests.intelligence.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.intelligence.vision.training.damage._dataset import prepared
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._framework import completed_fixture
from vision.damage.evaluation import EvaluationParams, run_evaluation


def test_threshold_change_is_not_selected_using_test(prepared, monkeypatch):
    config, manifest = prepared
    completed_fixture(config, manifest)
    monkeypatch.setenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.5")
    with pytest.raises(ValueError, match="never select it on TEST"):
        run_evaluation(
            config,
            params=EvaluationParams(
                dataset_loader=lambda _: manifest,
                model_factory=lambda _: pytest.fail("must not load"),
                environment_provider=lambda: {},
                max_error_examples=0,
            ),
        )
