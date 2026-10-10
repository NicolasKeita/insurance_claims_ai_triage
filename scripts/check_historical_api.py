"""Exercise HTTP history/investigation and verify the persisted evidence snapshot.

Default anomaly input is clearly labelled illustrative demo data. Optional
flags use an already trained local artifact or registered candidate; no training.
Each run appends an anomaly and an investigation assessment to the reference.
"""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from claims.assets import resolve_document_assets
from claims.consistency import compare_claim_documents
from claims.document_extraction import extract_accident_report, extract_claim_form, extract_garage_quote
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from persistence.repositories import AssessmentRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--registered-anomaly", action="store_true")
    source.add_argument("--anomaly-model", type=Path, help="Existing AnomalyDetector joblib artifact")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    case = load_claim_case(root / "data" / "CLAIM-2026-00001")
    assets = {asset.type: asset for asset in resolve_document_assets(case)}
    consistency = compare_claim_documents(
        extract_claim_form(extract_pdf_text(assets[DocumentType.CLAIM_FORM])),
        extract_accident_report(extract_pdf_text(assets[DocumentType.ACCIDENT_REPORT])),
        extract_garage_quote(extract_pdf_text(assets[DocumentType.REPAIR_QUOTE])),
    )

    if args.registered_anomaly:
        from ml.anomaly_inference import load_candidate_anomaly_predictor
        from ml.tracking import configure_mlflow_tracking

        configure_mlflow_tracking()
        loader = load_candidate_anomaly_predictor
        label = "Existing registered candidate anomaly model"
    else:
        detector = None
        if args.anomaly_model:
            import joblib

            detector = joblib.load(args.anomaly_model)
        label = ("Existing local anomaly model artifact (no training)" if detector
                 else "Illustrative demo anomaly input (not a model prediction)")

        class Predictor:
            def assess(self, features):
                assessment = (detector.assess(features) if detector else AnomalyAssessment(
                    is_anomalous=False, anomaly_score=0.42, threshold=0.62,
                    reference_percentile=0.5, statistical_signals=[],
                ))
                return RegisteredAnomalyAssessment(
                    assessment=assessment,
                    registered_model="local-anomaly-artifact" if detector else "demo-step25-current-anomaly-fixture",
                    model_version="existing-artifact" if detector else "demo-1",
                    model_alias="local-validation" if detector else "demo-fixture",
                )

        loader = Predictor

    def unused_triage():
        raise AssertionError("These endpoints must not load a triage model")

    engine = create_database_engine(DatabaseConfig.from_env())
    try:
        factory = create_session_factory(engine)
        app = create_app(unused_triage, loader, session_factory=factory)
        path = f"/v1/claims/{case.claim.claim_id}"
        with TestClient(app) as client:
            history = client.get(path + "/history")
            history.raise_for_status()
            assert client.get("/v1/claims/UNKNOWN-STEP25/history").status_code == 404
            response = client.post(path + "/investigation/assess",
                                   json={"consistency": consistency.model_dump(mode="json")})
            response.raise_for_status()
        body = response.json()
        assert body["policy_version"] == "investigation_policy_v2"
        with factory() as session:
            repository = AssessmentRepository(session)
            stored = repository.get_latest_investigation(case.claim.claim_id)
            anomaly = repository.get_latest_anomaly(case.claim.claim_id)
        assert stored.result.model_dump(mode="json") == body
        observed = {s.code.value: s.evidence for s in stored.result.signals
                    if s.category.value == "HISTORICAL_CONTEXT"}
        assert "RECENT_CLAIM_FREQUENCY" in observed
        assert "PREVIOUS_ANOMALOUS_CLAIMS" in observed
        print(f"Anomaly source: {label}")
        print(f"GET {path}/history: {history.status_code}")
        print(json.dumps(history.json(), indent=2))
        print("GET unknown history: 404")
        print(f"POST {path}/investigation/assess: {response.status_code}")
        print(json.dumps(body, indent=2))
        print(f"Persisted investigation: {stored.id}; policy={stored.result.policy_version}")
        print(f"Persisted anomaly: {anomaly.id}; model={anomaly.result.registered_model}")
        print("Historical signals persisted as exact evidence snapshots: verified")
        output = root / "artifacts" / "step25"
        output.mkdir(parents=True, exist_ok=True)
        (output / "history_response.json").write_text(json.dumps(history.json(), indent=2), encoding="utf-8")
        (output / "investigation_response.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
