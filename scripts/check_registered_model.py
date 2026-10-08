import os

import mlflow

from ml.features import TriageFeatures
from ml.inference import load_candidate_predictor
from ml.registry import CANDIDATE_ALIAS, REGISTERED_MODEL_NAME


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required")
    mlflow.set_tracking_uri(tracking_uri)
    predictor = load_candidate_predictor()
    features = TriageFeatures(
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
    print(f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}")
    print(predictor.predict(features).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
