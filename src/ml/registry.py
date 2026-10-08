from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

EXPERIMENT_NAME = "insurance-claims-triage"
REGISTERED_MODEL_NAME = "insurance-claims-triage-model"
CANDIDATE_ALIAS = "candidate"


@dataclass(frozen=True)
class CandidateModelSource:
    parent_run_id: str
    model_name: str
    model_uri: str
    dataset_version: str | None
    dataset_sha256: str | None
    git_commit: str | None


def candidate_source_from_run(run: Run) -> CandidateModelSource:
    name = run.data.tags.get("candidate.model")
    uri = run.data.tags.get("candidate.model_uri")
    if not name:
        raise ValueError("Run has no candidate.model tag")
    if not uri:
        raise ValueError("Run has no candidate.model_uri tag")
    return CandidateModelSource(
        parent_run_id=run.info.run_id,
        model_name=name,
        model_uri=uri,
        dataset_version=run.data.params.get("dataset_version"),
        dataset_sha256=run.data.params.get("dataset_sha256"),
        git_commit=run.data.tags.get("git.commit"),
    )


def find_latest_candidate_source(client: MlflowClient) -> CandidateModelSource:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME}")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=100,
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        if "candidate.model" in run.data.tags and "candidate.model_uri" in run.data.tags:
            return candidate_source_from_run(run)
    raise RuntimeError("No tracked candidate model found")


def register_candidate(*, client: MlflowClient, source: CandidateModelSource):
    version = mlflow.register_model(
        model_uri=source.model_uri,
        name=REGISTERED_MODEL_NAME,
    )
    tags = {
        "algorithm": source.model_name,
        "source.parent_run_id": source.parent_run_id,
        "dataset.version": source.dataset_version,
        "dataset.sha256": source.dataset_sha256,
        "git.commit": source.git_commit,
    }
    for key, value in tags.items():
        if value:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CANDIDATE_ALIAS, version.version
    )
    return version
