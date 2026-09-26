import pytest

from pydantic import ValidationError

from pathlib import Path

from claims import case
from dataclasses import FrozenInstanceError
from claims.loader import load_claim_case
from claims.enums import (
    ClaimType,
    DamageType,
    DocumentType,
)
from claims.validation import ClaimFileValidationError


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

    missing_documents = [
        document.type
        for document in claim.documents
        if not document.available
    ]

    assert DocumentType.POLICE_REPORT in missing_documents

def test_invalid_claim_is_rejected():
    claim_dir = Path("tests/data/invalid_claim")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_unknown_domain_value_is_rejected():
    claim_dir = Path("tests/data/invalid_claim_type")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_negative_repair_amount_is_rejected():
    claim_dir = Path("tests/data/invalid_negative_amount")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_invalid_vehicle_year_is_rejected():
    claim_dir = Path("tests/data/invalid_vehicle_year")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_available_document_without_filename_is_rejected():
    claim_dir = Path(
        "tests/data/invalid_available_document"
    )

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_invalid_currency_format_is_rejected():
    claim_dir = Path("tests/data/invalid_currency")

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_future_incident_is_rejected():
    claim_dir = Path(
        "tests/data/invalid_future_incident"
    )

    with pytest.raises(ValidationError):
        load_claim_case(claim_dir)

def test_missing_document_file_is_rejected():
    claim_dir = Path(
        "tests/data/missing_document"
    )

    with pytest.raises(ClaimFileValidationError):
        load_claim_case(claim_dir)

def test_missing_image_file_is_rejected():
    claim_dir = Path(
        "tests/data/missing_image"
    )

    with pytest.raises(ClaimFileValidationError):
        load_claim_case(claim_dir)

def test_claim_with_all_required_files_is_valid():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)
    claim = case.claim

    assert claim.claim_id == "CLAIM-2026-00001"

def test_loaded_claim_returns_case():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assert case.root_dir == claim_dir
    assert case.claim.claim_id == "CLAIM-2026-00001"

def test_claim_case_documents_directory():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assert case.documents_dir == (
        claim_dir / "documents"
    )

def test_claim_case_image_path():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    path = case.image_path("car_front.jpg")

    assert path == (
        claim_dir
        / "images"
        / "car_front.jpg"
    )

    assert path.is_file()

def test_claim_case_document_path():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    path = case.document_path(
        "garage_quote.pdf"
    )

    assert path == (
        claim_dir
        / "documents"
        / "garage_quote.pdf"
    )

    assert path.is_file()

def test_claim_case_root_directory_is_immutable():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    with pytest.raises(FrozenInstanceError):
        case.root_dir = Path("/tmp/other")