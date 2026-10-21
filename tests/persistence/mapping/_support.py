"""Support."""

from pathlib import Path

import pytest

from claims.loader import load_claim_case
from investigation.models import InvestigationAssessment, InvestigationSignal
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.inference import TriagePrediction


@pytest.fixture
def reference_claim():
    return load_claim_case(Path("data/CLAIM-2026-00001")).claim


def make_triage(version="1"):
    return TriagePrediction(
        recommended_workflow="STANDARD",
        confidence=0.7,
        probabilities={"STANDARD": 0.7, "FAST_TRACK": 0.3},
        registered_model="test-triage",
        model_version=version,
        model_alias="candidate",
    )


def make_anomaly(version="1"):
    return RegisteredAnomalyAssessment(
        assessment=AnomalyAssessment(
            is_anomalous=True,
            anomaly_score=0.72,
            threshold=0.62,
            reference_percentile=0.995,
            statistical_signals=["repair_amount: very_high"],
        ),
        registered_model="test-anomaly",
        model_version=version,
        model_alias="candidate",
    )


def make_investigation(version="investigation_policy_v1"):
    return InvestigationAssessment(
        review_score=18,
        review_priority="NORMAL",
        review_recommended=False,
        policy_version=version,
        signals=(
            InvestigationSignal(
                code="ANOMALOUS_PATTERN",
                category="ANOMALY_DETECTION",
                severity="MEDIUM",
                title="Observed atypicality",
                explanation="Unusual reference pattern",
                evidence={"anomaly_score": 0.72, "unusual_features": ["repair_amount: very_high"]},
            ),
        ),
    )
