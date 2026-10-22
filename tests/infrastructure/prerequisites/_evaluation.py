"""Offline boundaries for unit tests of the lightweight evaluation gate."""

import json

import pytest

from evaluation.models import EvaluationScenario
from evaluation.preflight import RealEvaluationConfig
from evaluation.preflight.prerequisites import PrerequisiteChecks
from evaluation.preflight.session import PreflightSession


def forbid_loading(*args, **kwargs):
    pytest.fail("Prerequisite checks must not load models or run inference")


@pytest.fixture
def gate(tmp_path, monkeypatch):
    for name in ("DATABASE_URL", "MLFLOW_TRACKING_URI", "CLAIMS_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "claim.json"
    source.write_text(json.dumps({"claim_id": "CONTROLLED-1"}), encoding="utf-8")
    checkpoint = tmp_path / "model.joblib"
    checkpoint.write_bytes(b"Not a serialized model; presence checks only")
    scenario = EvaluationScenario(
        scenario_id="controlled",
        description="Unit prerequisite contract",
        claim_source=str(source),
        infrastructure_requirements=["postgresql", "triage", "anomaly"],
        files=[str(source)],
        inputs={"claim_id": "CONTROLLED-1"},
    )
    config = RealEvaluationConfig(
        include_agent=False,
        include_rag=False,
        include_similar=False,
        include_vision=False,
        triage_checkpoint=checkpoint,
        anomaly_checkpoint=checkpoint,
    )
    for name in ("probe_embeddings", "probe_models", "probe_detectors", "probe_mlflow"):
        monkeypatch.setattr(PreflightSession, name, forbid_loading)
    return config, scenario


@pytest.fixture
def ready_gate(gate, monkeypatch):
    def postgres(session):
        session.probe("postgresql", lambda: {"schema_available": True})

    monkeypatch.setattr(PreflightSession, "probe_postgres", postgres)
    monkeypatch.setattr(PrerequisiteChecks, "stored_claim", lambda self, scenario: {})
    return gate
