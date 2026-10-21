"""Test loading."""

import ml.anomaly_inference as anomaly_inference
from ml.anomaly_inference import (
    load_candidate_anomaly_predictor,
)
from tests.intelligence.ml.anomaly.pyfunc._support import FakePyfuncModel, make_features


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
