import json

import pandas as pd
import pytest

import ml.anomaly_inference as anomaly_inference
from ml.anomaly_dataset import generate_anomaly_datasets
from ml.anomaly_inference import (
    RegisteredAnomalyPredictor,
    load_candidate_anomaly_predictor,
)
from ml.anomaly_pyfunc import AnomalyPyfuncModel
from ml.anomaly_training import ANOMALY_FEATURE_COLUMNS, train_anomaly_detector
from ml.features import TriageFeatures


def make_features() -> TriageFeatures:
    return TriageFeatures(
        claim_type="AUTO_COLLISION",
        collision_type="FRONT_COLLISION",
        vehicle_age=5,
        repair_amount=3160,
        declared_damage_count=2,
        additional_damage_count=1,
        document_count=4,
        missing_document_count=1,
        document_completeness_ratio=0.75,
        injuries_declared=False,
        consistency_issue_count=1,
    )


def test_pyfunc_uses_complete_detector_for_every_row() -> None:
    reference, evaluation = generate_anomaly_datasets(
        reference_count=100,
        evaluation_count=12,
        anomaly_fraction=0.25,
        seed=42,
    )
    detector = train_anomaly_detector(reference, n_estimators=20)
    model = AnomalyPyfuncModel(detector)
    inputs = evaluation.iloc[[1, 4, 7]].copy()
    result = model.predict(None, inputs)

    assert result.index.equals(inputs.index)
    assert list(result.columns) == [
        "is_anomalous",
        "anomaly_score",
        "threshold",
        "reference_percentile",
        "statistical_signals",
    ]
    for position, (_, claim) in enumerate(inputs.iterrows()):
        expected = detector.assess(claim)
        actual = result.iloc[position]
        assert bool(actual["is_anomalous"]) == expected.is_anomalous
        assert actual["anomaly_score"] == pytest.approx(expected.anomaly_score)
        assert actual["threshold"] == pytest.approx(expected.threshold)
        assert actual["reference_percentile"] == pytest.approx(
            expected.reference_percentile
        )
        assert json.loads(actual["statistical_signals"]) == expected.statistical_signals


class FakePyfuncModel:
    def __init__(self) -> None:
        self.input_frame: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.input_frame = frame
        return pd.DataFrame(
            [
                {
                    "is_anomalous": True,
                    "anomaly_score": 0.72,
                    "threshold": 0.61,
                    "reference_percentile": 0.995,
                    "statistical_signals": '["repair_amount: very_high"]',
                }
            ]
        )


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


def test_loader_resolves_candidate_alias_without_server(monkeypatch) -> None:
    calls = {}
    model = FakePyfuncModel()

    class FakeVersion:
        version = "11"

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str):
            calls["registry"] = (name, alias)
            return FakeVersion()

    def fake_load_model(uri: str):
        calls["uri"] = uri
        return model

    monkeypatch.setattr(anomaly_inference, "MlflowClient", FakeClient)
    monkeypatch.setattr(anomaly_inference.mlflow.pyfunc, "load_model", fake_load_model)

    predictor = load_candidate_anomaly_predictor()

    assert calls["registry"] == ("insurance-claims-anomaly-model", "candidate")
    assert calls["uri"] == "models:/insurance-claims-anomaly-model@candidate"
    assert predictor.version == "11"
    assert predictor.assess(make_features()).model_version == "11"
