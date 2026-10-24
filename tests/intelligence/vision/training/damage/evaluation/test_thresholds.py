"""Test thresholds."""

import pytest

from tests.intelligence.vision.training.damage._dataset import (
    offline_environment as offline_environment,
)
from tests.intelligence.vision.training.damage._dataset import prepared
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._framework import OrchestrationStub, completed_fixture
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


def test_ap_floor_is_independent_from_recorded_operational_threshold(prepared):
    config, manifest = prepared
    completed_fixture(config, manifest)
    model = OrchestrationStub()
    metrics = run_evaluation(
        config,
        params=EvaluationParams(
            model_factory=lambda _: model,
            environment_provider=lambda: {},
            dataset_loader=lambda _: manifest,
            max_error_examples=0,
            ap_confidence=0.001,
        ),
    )
    assert model.validation_kwargs["conf"] == 0.001
    assert metrics["ap_confidence"] == 0.001
    assert metrics["confidence_threshold"] == 0.25
    assert metrics["schema_version"] == 2


@pytest.mark.parametrize("floor", [-0.1, 0.3, float("nan"), float("inf"), True, "0.001"])
def test_invalid_ap_floor_fails_before_model_loading(prepared, floor):
    config, manifest = prepared
    completed_fixture(config, manifest)
    with pytest.raises(ValueError, match="AP confidence"):
        run_evaluation(
            config,
            params=EvaluationParams(
                model_factory=lambda _: pytest.fail("must not load"),
                environment_provider=lambda: pytest.fail("must not import"),
                dataset_loader=lambda _: manifest,
                max_error_examples=0,
                ap_confidence=floor,
            ),
        )
