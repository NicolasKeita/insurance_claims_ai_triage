from ml.inference import RegisteredTriagePredictor
from tests.ml.triage._inference_fakes import FakeClassifier as FakeClassifier
from tests.ml.triage._inference_fakes import FakePipeline as FakePipeline
from tests.ml.triage._inference_fakes import make_features as make_features


def test_predictor_returns_probabilities():
    predictor = RegisteredTriagePredictor(FakePipeline(), "test-model", "7", "candidate")
    result = predictor.predict(make_features())
    assert result.recommended_workflow == "STANDARD"
    assert result.confidence == 0.70
    assert result.probabilities["STANDARD"] == 0.70
    assert result.model_version == "7"
