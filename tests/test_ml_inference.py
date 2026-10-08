import numpy as np

from ml.features import TriageFeatures
from ml.inference import RegisteredTriagePredictor
from ml.training import FEATURE_COLUMNS


class FakeClassifier:
    classes_ = np.array(["EXPERT_REVIEW", "FAST_TRACK", "INVESTIGATION", "STANDARD"])


class FakePipeline:
    def __init__(self):
        self.named_steps = {"classifier": FakeClassifier()}

    def predict(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return np.array(["STANDARD"])

    def predict_proba(self, frame):
        return np.array([[0.10, 0.15, 0.05, 0.70]])


def make_features():
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


def test_predictor_returns_probabilities():
    predictor = RegisteredTriagePredictor(FakePipeline(), "test-model", "7", "candidate")
    result = predictor.predict(make_features())
    assert result.recommended_workflow == "STANDARD"
    assert result.confidence == 0.70
    assert result.probabilities["STANDARD"] == 0.70
    assert result.model_version == "7"
