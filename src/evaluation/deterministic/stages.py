"""Deterministic run state and the claim, document and feature stages."""

from dataclasses import dataclass, field
from hashlib import sha256

from claims.enums import DocumentType
from evaluation.deterministic.documents import extract_case_documents
from evaluation.deterministic.fixtures import SYNTHETIC_PROVENANCE, materialize_scenario_case
from evaluation.deterministic.recording import TraceRecorder
from evaluation.models import PipelineObservation


@dataclass
class DeterministicRun:
    """Mutable state carried across the deterministic stages of one scenario."""

    scenario: object
    output: PipelineObservation
    recorder: TraceRecorder
    triage_predictor: object = None
    anomaly_predictor: object = None
    case: object = None
    extracted: dict = field(default_factory=dict)
    ingested: dict = field(default_factory=dict)
    form: object = None
    report: object = None
    quote: object = None
    consistency: object = None
    features: object = None
    triage: object = None
    anomaly: object = None
    history: object = None
    history_inputs: list = field(default_factory=list)
    visual: object = None
    similar: object = None
    retriever: object = None
    embeddings: object = None
    llm: object = None


def load_claim(run, workspace):
    """Materialize the controlled claim fixture and record the claim snapshot."""
    run.case = run.recorder.record(
        "claim",
        lambda: materialize_scenario_case(run.scenario, workspace),
        inputs={"claim_source": run.scenario.claim_source},
    )
    run.output.facts["claim"] = run.case.claim.model_dump(mode="json")


def extract_documents(run):
    """Ingest and structure the created documents, recording hashes and sources."""
    case = run.case
    run.extracted, run.ingested = run.recorder.record(
        "document",
        lambda: extract_case_documents(case),
        inputs={
            "claim_id": case.claim.claim_id,
            "fixture_source": run.scenario.claim_source,
            "document_entries": [entry.model_dump(mode="json") for entry in case.claim.documents],
        },
        sources=[entry.filename for entry in case.claim.documents if entry.available],
    )
    run.form, run.report, run.quote = (
        run.extracted[kind]
        for kind in (DocumentType.CLAIM_FORM, DocumentType.ACCIDENT_REPORT, DocumentType.REPAIR_QUOTE)
    )
    run.output.facts["extraction"] = {
        "success": True,
        "form": run.form.model_dump(mode="json"),
        "report": run.report.model_dump(mode="json"),
        "quote": run.quote.model_dump(mode="json"),
    }
    run.output.provenance["documents"] = [
        {
            **item.source.model_dump(mode="json"),
            "asset_sha256": sha256(run.ingested[item.source.filename].asset.path.read_bytes()).hexdigest(),
            "text_sha256": sha256(run.ingested[item.source.filename].text.encode("utf-8")).hexdigest(),
            "fixture_source": run.scenario.claim_source,
            "synthetic": True,
        }
        for item in (run.form, run.report, run.quote)
    ]
    run.output.provenance["fixture"] = {"kind": "controlled-synthetic", "description": SYNTHETIC_PROVENANCE}


def compare_documents(run):
    """Compare the parsed documents with the production consistency service."""
    from claims.consistency import compare_claim_documents

    run.consistency = run.recorder.record(
        "consistency", lambda: compare_claim_documents(run.form, run.report, run.quote)
    )
    run.output.facts["consistency"] = {
        **run.consistency.model_dump(mode="json"),
        "is_consistent": run.consistency.is_consistent,
    }


def build_features(run):
    """Build the production triage features from the compared documents."""
    from ml.features import build_triage_features

    run.features = run.recorder.record(
        "features", lambda: build_triage_features(run.case, run.form, run.quote, run.consistency)
    )
    run.output.facts["features"] = run.features.model_dump(mode="json")


def assess_history(run):
    """Aggregate controlled history through the real repository contract."""
    from evaluation.fakes import FakeHistoricalRepository, historical_claims

    run.history_inputs = historical_claims(run.case.claim, run.scenario.inputs.get("history_claims", []))
    run.history = run.recorder.record(
        "history",
        lambda: FakeHistoricalRepository(run.case.claim, run.history_inputs).get_profile(run.case.claim.claim_id),
    )
    run.output.facts["history"] = run.history.model_dump(mode="json")