import pytest

from pydantic import ValidationError

from pathlib import Path

from claims.loader import load_claim


def test_load_claim():
    claim_path = Path("data/CLAIM-2026-00001/claim.json")

    claim = load_claim(claim_path)

    assert claim.claim_id == "CLAIM-2026-00001"
    assert claim.claim_type == "AUTO_COLLISION"

    assert claim.vehicle.make == "Renault"
    assert claim.vehicle.model == "Clio"
    assert claim.vehicle.year == 2021

    assert claim.repair_estimate.amount == 3160
    assert claim.repair_estimate.currency == "EUR"

def test_claim_contains_missing_police_report():
    claim_path = Path("data/CLAIM-2026-00001/claim.json")

    claim = load_claim(claim_path)

    missing_documents = [
        document.type
        for document in claim.documents
        if not document.available
    ]

    assert "POLICE_REPORT" in missing_documents

def test_invalid_claim_is_rejected():
    claim_path = Path("tests/data/invalid_claim.json")

    with pytest.raises(ValidationError):
        load_claim(claim_path)