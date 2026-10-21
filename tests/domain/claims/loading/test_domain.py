"""Test domain."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from claims.enums import (
    ClaimType,
    DamageType,
    DocumentType,
)
from claims.loader import load_claim_case


def test_load_claim():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)
    claim = case.claim

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

    case = load_claim_case(claim_dir)
    claim = case.claim

    missing_documents = [document.type for document in claim.documents if not document.available]

    assert DocumentType.POLICE_REPORT in missing_documents


def test_invalid_claim_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_claim")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)


def test_unknown_domain_value_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_claim_type")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)


def test_negative_repair_amount_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_negative_amount")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)


def test_invalid_vehicle_year_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_vehicle_year")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)


def test_available_document_without_filename_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_available_document")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)


def test_invalid_currency_format_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_currency")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)


def test_future_incident_is_rejected():
    claim_dir = Path("tests/data/invalid/invalid_future_incident")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)
