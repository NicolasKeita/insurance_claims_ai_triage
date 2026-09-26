import pytest

from pydantic import ValidationError

from pathlib import Path

from claims.loader import load_claim
from claims.enums import (
    ClaimType,
    DamageType,
    DocumentType,
)


def test_load_claim():
    claim_dir = Path("data/CLAIM-2026-00001")

    claim = load_claim(claim_dir)

    assert claim.claim_id == "CLAIM-2026-00001"
    assert claim.claim_type == ClaimType.AUTO_COLLISION

    assert claim.vehicle.make == "Renault"
    assert claim.vehicle.model == "Clio"
    assert claim.vehicle.year == 2021

    assert claim.repair_estimate.amount == 3160
    assert claim.repair_estimate.currency == "EUR"

    assert claim.declared_damage == [
        DamageType.FRONT_BUMPER,
        DamageType.LEFT_HEADLIGHT,
    ]

def test_claim_contains_missing_police_report():
    claim_dir = Path("data/CLAIM-2026-00001")

    claim = load_claim(claim_dir)

    missing_documents = [
        document.type
        for document in claim.documents
        if not document.available
    ]

    assert DocumentType.POLICE_REPORT in missing_documents

def test_invalid_claim_is_rejected():
    claim_path = Path("tests/data/invalid_claim.json")

    with pytest.raises(ValidationError):
        load_claim(claim_path)

def test_unknown_domain_value_is_rejected():
    claim_path = Path("tests/data/invalid_claim_type.json")

    with pytest.raises(ValidationError):
        load_claim(claim_path)

def test_negative_repair_amount_is_rejected():
    claim_path = Path("tests/data/invalid_negative_amount.json")

    with pytest.raises(ValidationError):
        load_claim(claim_path)

def test_invalid_vehicle_year_is_rejected():
    claim_path = Path("tests/data/invalid_vehicle_year.json")

    with pytest.raises(ValidationError):
        load_claim(claim_path)

def test_available_document_without_filename_is_rejected():
    claim_path = Path(
        "tests/data/invalid_available_document.json"
    )

    with pytest.raises(ValidationError):
        load_claim(claim_path)

def test_invalid_currency_format_is_rejected():
    claim_path = Path("tests/data/invalid_currency.json")

    with pytest.raises(ValidationError):
        load_claim(claim_path)

def test_future_incident_is_rejected():
    claim_path = Path(
        "tests/data/invalid_future_incident.json"
    )

    with pytest.raises(ValidationError):
        load_claim(claim_path)