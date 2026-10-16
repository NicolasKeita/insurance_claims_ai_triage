"""Stable predictor stand-ins for deterministic workflow observations."""

from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.inference import TriagePrediction


class FakeTriagePredictor:
    """A stable workflow observation, without claiming a business ground truth."""

    def predict(self, features):
        return TriagePrediction(
            recommended_workflow="STANDARD",
            confidence=0.75,
            probabilities={"STANDARD": 0.75, "EXPERT_REVIEW": 0.25},
            registered_model="fake-triage-e2e",
            model_version="test-only-v1",
            model_alias="fixture",
        )


class FakeAnomalyPredictor:
    """A stable statistical observation, never a fraud determination."""

    def __init__(self, fixture=None):
        self.fixture = fixture or {}

    def assess(self, features):
        values = {
            "is_anomalous": False,
            "anomaly_score": 0.3,
            "threshold": 0.6,
            "reference_percentile": 0.5,
            "statistical_signals": [],
            **self.fixture,
        }
        return RegisteredAnomalyAssessment(
            assessment=AnomalyAssessment.model_validate(values),
            registered_model="fake-anomaly-e2e",
            model_version="test-only-v1",
            model_alias="fixture",
        )