"""Register the latest successfully tracked anomaly PyFunc as candidate."""

from mlflow import MlflowClient

from ml.anomaly_registry import (
    ANOMALY_CANDIDATE_ALIAS,
    REGISTERED_ANOMALY_MODEL_NAME,
    find_latest_anomaly_candidate_source,
    register_anomaly_candidate,
)
from ml.tracking import configure_mlflow_tracking


def main() -> None:
    configure_mlflow_tracking()
    client = MlflowClient()
    source = find_latest_anomaly_candidate_source(client)
    version = register_anomaly_candidate(client=client, source=source)
    print(f"Registered model: {REGISTERED_ANOMALY_MODEL_NAME}")
    print(f"Version: {version.version}")
    print(f"Alias: {ANOMALY_CANDIDATE_ALIAS}")
    print(f"Source run: {source.run_id}")
    print(f"Threshold: {source.threshold_value}")
    print(f"Logged model URI: {source.model_uri}")
    print(
        "Registry model URI: "
        f"models:/{REGISTERED_ANOMALY_MODEL_NAME}@{ANOMALY_CANDIDATE_ALIAS}"
    )


if __name__ == "__main__":
    main()
