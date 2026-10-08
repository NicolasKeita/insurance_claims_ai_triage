import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.registry import (
    CANDIDATE_ALIAS,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    find_latest_candidate_source,
    register_candidate,
)


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit(
            "Erreur : MLFLOW_TRACKING_URI n'est pas défini. "
            "Dans PowerShell : $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'"
        )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        source = find_latest_candidate_source(client)
        print(f"Tracked candidate: {source.model_name} ({source.model_uri})")
        print(f"Parent run: {source.parent_run_id}")
        version = register_candidate(client=client, source=source)
    except MlflowException as exc:
        if "Failed to establish a new connection" not in str(exc):
            raise
        raise SystemExit(
            f"Erreur : impossible de joindre MLflow sur {tracking_uri}.\n"
            "Lance MLflow dans un autre terminal PowerShell avec : "
            "python -m mlflow server --host 127.0.0.1 --port 5000"
        ) from None
    except RuntimeError as exc:
        if str(exc) == f"MLflow experiment not found: {EXPERIMENT_NAME}":
            raise SystemExit(
                f"Erreur : l'experience MLflow '{EXPERIMENT_NAME}' n'existe pas "
                f"encore sur {tracking_uri}.\n"
                "Lance d'abord : python .\\scripts\\track_triage_experiment.py\n"
                "Puis relance : python .\\scripts\\register_triage_candidate.py"
            ) from None
        if str(exc) == "No tracked candidate model found":
            raise SystemExit(
                f"Erreur : aucun modele candidat dans l'experience '{EXPERIMENT_NAME}'.\n"
                "Lance d'abord : python .\\scripts\\track_triage_experiment.py\n"
                "Puis relance : python .\\scripts\\register_triage_candidate.py"
            ) from None
        raise
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {version.version}; alias: {CANDIDATE_ALIAS}")


if __name__ == "__main__":
    main()
