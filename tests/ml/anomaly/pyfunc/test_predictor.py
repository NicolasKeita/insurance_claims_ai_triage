"""Test predictor."""

import pandas as pd
import pytest

from ml.anomaly_inference import (
    RegisteredAnomalyPredictor,
)
from ml.anomaly_training import ANOMALY_FEATURE_COLUMNS
from tests.ml.anomaly.pyfunc._support import FakePyfuncModel, make_features


def test_registered_predictor_converts_features_and_result() -> None:
    model = FakePyfuncModel()
    predictor = RegisteredAnomalyPredictor(
        model=model,
        registered_model="insurance-claims-anomaly-model",
        version="7",
        alias="candidate",
    )

    result = predictor.assess(make_features())

    assert model.input_frame is not None
    assert list(model.input_frame.columns) == list(ANOMALY_FEATURE_COLUMNS)
    assert model.input_frame.iloc[0].to_dict() == make_features().model_dump(mode="json")
    assert result.assessment.is_anomalous
    assert result.assessment.anomaly_score == 0.72
    assert result.assessment.threshold == 0.61
    assert result.assessment.reference_percentile == 0.995
    assert result.assessment.statistical_signals == ["repair_amount: very_high"]
    assert result.registered_model == "insurance-claims-anomaly-model"
    assert result.model_version == "7"
    assert result.model_alias == "candidate"


def test_registered_predictor_rejects_malformed_model_output() -> None:
    class BadModel:
        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame([{"statistical_signals": "not JSON"}])

    predictor = RegisteredAnomalyPredictor(BadModel(), "test", "1", "candidate")
    with pytest.raises(ValueError):
        predictor.assess(make_features())
