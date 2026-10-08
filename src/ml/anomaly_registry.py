"""Model Registry operations for the complete novelty detector."""

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.entities import Run

from ml.registry import CANDIDATE_ALIAS, register_model_version_with_alias

ANOMALY_EXPERIMENT_NAME = "insurance-claims-anomaly"
REGISTERED_ANOMALY_MODEL_NAME = "insurance-claims-anomaly-model"
ANOMALY_CANDIDATE_ALIAS = CANDIDATE_ALIAS


@dataclass(frozen=True)
class AnomalyCandidateSource:
    run_id: str
    model_uri: str
    reference_version: str
    reference_sha256: str
    evaluation_version: str
    evaluation_sha256: str
    threshold_percentile: str
    threshold_value: str
    git_commit: str | None


def anomaly_candidate_source_from_run(run: Run) -> AnomalyCandidateSource:
    """Extract the lineage required to register one finished anomaly run."""
    if run.info.status != "FINISHED":
        raise ValueError("Candidate run is not finished")
    if run.data.tags.get("task") != "novelty_detection":
        raise ValueError("Candidate run is not a novelty detection run")

    values = {
        "model_uri": run.data.tags.get("candidate.model_uri"),
        "reference_version": run.data.params.get("dataset.reference.version"),
        "reference_sha256": run.data.params.get("dataset.reference.sha256"),
        "evaluation_version": run.data.params.get("dataset.evaluation.version"),
        "evaluation_sha256": run.data.params.get("dataset.evaluation.sha256"),
        "threshold_percentile": run.data.params.get("threshold.percentile"),
        "threshold_value": run.data.params.get("threshold.value"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError("Candidate run is missing: " + ", ".join(missing))

    return AnomalyCandidateSource(
        run_id=run.info.run_id,
        git_commit=run.data.tags.get("git.commit"),
        **values,
    )


def find_latest_anomaly_candidate_source(client: MlflowClient) -> AnomalyCandidateSource:
    """Find the newest eligible run, paging past unrelated or incomplete runs."""
    experiment = client.get_experiment_by_name(ANOMALY_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {ANOMALY_EXPERIMENT_NAME}")

    page_token = None
    while True:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100,
            order_by=["attributes.start_time DESC"],
            page_token=page_token,
        )
        for run in runs:
            try:
                return anomaly_candidate_source_from_run(run)
            except ValueError:
                continue
        page_token = getattr(runs, "token", None)
        if not page_token:
            break
    raise RuntimeError("No tracked anomaly candidate model found")


def register_anomaly_candidate(
    *, client: MlflowClient, source: AnomalyCandidateSource
):
    """Register a logged PyFunc model and update the candidate alias."""
    return register_model_version_with_alias(
        client=client,
        model_uri=source.model_uri,
        name=REGISTERED_ANOMALY_MODEL_NAME,
        alias=ANOMALY_CANDIDATE_ALIAS,
        tags={
            "algorithm": "IsolationForest",
            "task": "novelty_detection",
            "dataset.reference.version": source.reference_version,
            "dataset.reference.sha256": source.reference_sha256,
            "dataset.evaluation.version": source.evaluation_version,
            "dataset.evaluation.sha256": source.evaluation_sha256,
            "threshold.percentile": source.threshold_percentile,
            "threshold.value": source.threshold_value,
            "git.commit": source.git_commit,
            "source.run_id": source.run_id,
        },
    )
