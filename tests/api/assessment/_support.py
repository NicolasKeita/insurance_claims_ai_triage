"""Support."""

from claims.consistency import ClaimConsistencyReport
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.inference import RegisteredTriagePredictor
from tests.ml.triage._inference_fakes import FakePipeline, make_features


def fake_predictor_loader():
    return RegisteredTriagePredictor(FakePipeline(), "test-model", "1", "candidate")


class FakeAnomalyPredictor:
    registered_model = "test-anomaly-model"
    version = "2"
    alias = "candidate"

    def assess(self, features):
        assert features in (make_features(), make_combined_features())
        return RegisteredAnomalyAssessment(
            assessment=AnomalyAssessment(
                is_anomalous=True,
                anomaly_score=0.72,
                threshold=0.62,
                reference_percentile=0.995,
                statistical_signals=["repair_amount: very_high"],
            ),
            registered_model=self.registered_model,
            model_version=self.version,
            model_alias=self.alias,
        )


def fake_anomaly_predictor_loader():
    return FakeAnomalyPredictor()


def make_consistency():
    return ClaimConsistencyReport(
        claim_id_match=True,
        incident_date_match=False,
        location_match=True,
        vehicle_match=True,
        collision_type_match=True,
        injuries_match=True,
        declared_damage_match=True,
        additional_quote_damage=("HOOD",),
        unmapped_quote_items=("paint material",),
    )


def make_combined_features():
    return make_features().model_copy(update={"consistency_issue_count": 3})


def make_combined_payload():
    return {
        "features": make_combined_features().model_dump(mode="json"),
        "consistency": make_consistency().model_dump(mode="json"),
    }
