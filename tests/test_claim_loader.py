import pytest

from pydantic import ValidationError

from pathlib import Path

from claims.loader import load_claim
from claims.enums import (
    ClaimType,
    DamageType,
    DocumentType,
)
from claims.validation import ClaimFileValidationError


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
    claim_dir = Path("tests/data/invalid_claim")

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_unknown_domain_value_is_rejected():
    claim_dir = Path("tests/data/invalid_claim_type")

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_negative_repair_amount_is_rejected():
    claim_dir = Path("tests/data/invalid_negative_amount")

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_invalid_vehicle_year_is_rejected():
    claim_dir = Path("tests/data/invalid_vehicle_year")

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_available_document_without_filename_is_rejected():
    claim_dir = Path(
        "tests/data/invalid_available_document"
    )

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_invalid_currency_format_is_rejected():
    claim_dir = Path("tests/data/invalid_currency")

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_future_incident_is_rejected():
    claim_dir = Path(
        "tests/data/invalid_future_incident"
    )

    with pytest.raises(ValidationError):
        load_claim(claim_dir)

def test_missing_document_file_is_rejected():
    claim_dir = Path(
        "tests/data/missing_document"
    )

    with pytest.raises(ClaimFileValidationError):
        load_claim(claim_dir)

def test_missing_image_file_is_rejected():
    claim_dir = Path(
        "tests/data/missing_image"
    )

    with pytest.raises(ClaimFileValidationError):
        load_claim(claim_dir)

def test_claim_with_all_required_files_is_valid():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    claim = load_claim(claim_dir)
    assert claim.claim_id == "CLAIM-2026-00001"