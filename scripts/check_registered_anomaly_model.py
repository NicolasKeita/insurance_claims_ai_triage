"""Smoke test normal and atypical claims against the registered anomaly model."""

from ml.anomaly_inference import load_candidate_anomaly_predictor
from ml.features import TriageFeatures
from ml.tracking import configure_mlflow_tracking


def main() -> None:
    configure_mlflow_tracking()
    predictor = load_candidate_anomaly_predictor()
    ordinary = TriageFeatures(
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
    atypical = TriageFeatures(
        claim_type="AUTO_COLLISION",
        collision_type="PARKING_DAMAGE",
        vehicle_age=35,
        repair_amount=29000,
        declared_damage_count=8,
        additional_damage_count=4,
        document_count=6,
        missing_document_count=5,
        document_completeness_ratio=1 / 6,
        injuries_declared=False,
        consistency_issue_count=8,
    )
    print(f"Registered model: {predictor.registered_model}")
    print(f"Version: {predictor.version}")
    print(f"Alias: {predictor.alias}")
    print(f"Normal: {predictor.assess(ordinary).model_dump_json(indent=2)}")
    print(f"Atypical: {predictor.assess(atypical).model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
