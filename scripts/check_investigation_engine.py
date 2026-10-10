"""Demonstrate deterministic investigation signals for the reference claim.

The default anomaly assessment is an illustrative structured input so the
script works without MLflow. Use --registered-anomaly to score the claim with
the registered candidate model when a tracking server is available.
"""

import argparse
import json
from pathlib import Path

from claims.assets import resolve_document_assets
from claims.consistency import compare_claim_documents
from claims.document_extraction import (
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
)
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case
from investigation import assess_investigation
from ml.anomaly_training import AnomalyAssessment
from ml.features import build_triage_features


REFERENCE_CLAIM = Path(__file__).resolve().parents[1] / "data" / "CLAIM-2026-00001"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registered-anomaly",
        action="store_true",
        help="Use the MLflow candidate anomaly model (requires MLFLOW_TRACKING_URI).",
    )
    args = parser.parse_args()

    case = load_claim_case(REFERENCE_CLAIM)
    assets = {asset.type: asset for asset in resolve_document_assets(case)}
    form = extract_claim_form(extract_pdf_text(assets[DocumentType.CLAIM_FORM]))
    report = extract_accident_report(
        extract_pdf_text(assets[DocumentType.ACCIDENT_REPORT])
    )
    quote = extract_garage_quote(extract_pdf_text(assets[DocumentType.REPAIR_QUOTE]))
    consistency = compare_claim_documents(form, report, quote)
    features = build_triage_features(case, form, quote, consistency)

    if args.registered_anomaly:
        from ml.anomaly_inference import load_candidate_anomaly_predictor
        from ml.tracking import configure_mlflow_tracking

        configure_mlflow_tracking()
        anomaly = load_candidate_anomaly_predictor().assess(features).assessment
        anomaly_source = "registered candidate anomaly model"
    else:
        anomaly = AnomalyAssessment(
            is_anomalous=False,
            anomaly_score=0.42,
            threshold=0.62,
            reference_percentile=0.75,
            statistical_signals=[],
        )
        anomaly_source = "illustrative local input (not a model prediction)"

    assessment = assess_investigation(
        consistency=consistency,
        anomaly=anomaly,
        features=features,
    )
    print(f"Claim: {case.claim.claim_id}")
    print(f"Anomaly source: {anomaly_source}")
    print(f"Policy version: {assessment.policy_version}")
    print(f"Review score: {assessment.review_score}")
    print(f"Review priority: {assessment.review_priority.value}")
    print(f"Review recommended: {assessment.review_recommended}")
    for signal in assessment.signals:
        print(
            f"\n{signal.code.value} "
            f"[{signal.category.value} / {signal.severity.value}]"
        )
        print(f"  {signal.explanation}")
        print(f"  Evidence: {json.dumps(signal.evidence, ensure_ascii=False, sort_keys=True)}")
    print("\nInvestigationAssessment JSON:")
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
