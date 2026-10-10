"""Pure mapping/configuration tests: no database, Docker or inference server."""

import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from claims.loader import load_claim_case
from investigation.models import InvestigationAssessment, InvestigationSignal
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.inference import TriagePrediction
from persistence.database import DatabaseConfig, DatabaseConfigurationError, require_test_database
from persistence.mappers import (
    anomaly_to_row, checked_triage, claim_to_row, investigation_to_row,
    row_to_claim, triage_to_row,
)
from persistence.models import CustomerRow, PolicyRow, utc_now


@pytest.fixture
def reference_claim():
    return load_claim_case(Path("data/CLAIM-2026-00001")).claim


def make_triage(version="1"):
    return TriagePrediction(
        recommended_workflow="STANDARD", confidence=0.7,
        probabilities={"STANDARD": 0.7, "FAST_TRACK": 0.3},
        registered_model="test-triage", model_version=version, model_alias="candidate",
    )


def make_anomaly(version="1"):
    return RegisteredAnomalyAssessment(
        assessment=AnomalyAssessment(is_anomalous=True, anomaly_score=0.72,
                                     threshold=0.62, reference_percentile=0.995,
                                     statistical_signals=["repair_amount: very_high"]),
        registered_model="test-anomaly", model_version=version, model_alias="candidate",
    )


def make_investigation(version="investigation_policy_v1"):
    return InvestigationAssessment(
        review_score=18, review_priority="NORMAL", review_recommended=False,
        policy_version=version,
        signals=(InvestigationSignal(
            code="ANOMALOUS_PATTERN", category="ANOMALY_DETECTION", severity="MEDIUM",
            title="Observed atypicality", explanation="Unusual reference pattern",
            evidence={"anomaly_score": 0.72, "unusual_features": ["repair_amount: very_high"]},
        ),),
    )


def test_claim_mapping_round_trip_preserves_domain(reference_claim):
    row = claim_to_row(reference_claim,
                       CustomerRow(customer_id=reference_claim.customer.customer_id),
                       PolicyRow(policy_id=reference_claim.policy.policy_id,
                                 product=reference_claim.policy.product))
    assert row_to_claim(row) == reference_claim
    assert isinstance(row.repair_estimate_amount, Decimal)
    assert len(row.documents) == 4
    assert len(row.images) == 2
    assert len(row.damages) == 2
    assert row.documents[-1].filename is None
    assert "id" not in type(row_to_claim(row)).model_fields


def test_loader_preserves_decimal_json_precision(tmp_path, reference_claim):
    payload = reference_claim.model_dump(mode="json")
    payload.update(documents=[], images=[])
    text = json.dumps(payload).replace('"3160"', '9007199254740993.12345678')
    (tmp_path / "claim.json").write_text(text, encoding="utf-8")
    assert load_claim_case(tmp_path).claim.repair_estimate.amount == Decimal("9007199254740993.12345678")


def test_duplicate_damage_rejected_before_writes(reference_claim):
    claim = reference_claim.model_copy(update={"declared_damage": reference_claim.declared_damage * 2})
    with pytest.raises(ValueError, match="only once"):
        claim_to_row(claim, CustomerRow(customer_id="c"), PolicyRow(policy_id="p", product="a"))


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity"])
def test_invalid_financial_values_rejected(reference_claim, amount):
    payload = reference_claim.model_dump()
    payload["repair_estimate"]["amount"] = amount
    with pytest.raises(ValidationError):
        type(reference_claim).model_validate(payload)


def test_assessment_mappers_keep_metadata_and_evidence():
    pk = uuid4()
    triage = triage_to_row(pk, make_triage())
    anomaly = anomaly_to_row(pk, make_anomaly())
    source = make_investigation()
    investigation = investigation_to_row(pk, source)
    assert triage.probabilities == make_triage().probabilities
    assert anomaly.unusual_features == make_anomaly().assessment.statistical_signals
    assert anomaly.reference_percentile == 0.995
    assert investigation.signals == source.model_dump(mode="json")["signals"]
    source.signals[0].evidence["unusual_features"].append("changed later")
    assert investigation.signals[0]["evidence"]["unusual_features"] == ["repair_amount: very_high"]
    json.dumps(investigation.signals, allow_nan=False)


@pytest.mark.parametrize("probabilities", [
    {}, {"STANDARD": 1.1, "FAST_TRACK": -0.1},
    {"STANDARD": 0.7, "FAST_TRACK": 0.1}, {"OTHER": 1},
    {"STANDARD": float("nan")}, {"STANDARD": 0.8, "FAST_TRACK": 0.2},
])
def test_incoherent_probabilities_rejected(probabilities):
    with pytest.raises(ValueError, match="Incoherent"):
        checked_triage(make_triage().model_copy(update={"probabilities": probabilities}))


def test_nonfinite_anomaly_rejected():
    result = make_anomaly()
    result.assessment.anomaly_score = float("nan")
    with pytest.raises(ValueError, match="finite"):
        anomaly_to_row(uuid4(), result)


def test_database_configuration_is_explicit(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigurationError, match="required"):
        DatabaseConfig.from_env()
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_url("sqlite://")
    assert utc_now().utcoffset().total_seconds() == 0


@pytest.mark.parametrize("database", ["insurance_claims", "production", "test", "insurance_claims_test_backup"])
def test_destructive_test_guard_rejects_unsafe_names(database, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigurationError, match="_test"):
        require_test_database(f"postgresql+psycopg://localhost/{database}")


def test_test_database_must_differ_from_development(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/insurance_claims_test")
    with pytest.raises(DatabaseConfigurationError, match="differ"):
        require_test_database("postgresql+psycopg://localhost:5432/insurance_claims_test")
