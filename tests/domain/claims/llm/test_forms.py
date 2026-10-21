"""Test forms."""

from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.enums import DamageType, DocumentType
from claims.llm_document_extraction import (
    extract_accident_report_with_llm,
    extract_claim_form_with_llm,
)
from claims.loader import load_claim_case
from tests.domain.claims.llm._llm import FakeStructuredLlm


def test_extract_claim_form_with_llm():
    case = load_claim_case(Path("data/CLAIM-2026-00001"))

    assets = resolve_document_assets(case)

    asset = next(asset for asset in assets if asset.type == DocumentType.CLAIM_FORM)

    extraction = extract_pdf_text(asset)

    llm = FakeStructuredLlm(
        {
            "claim_id": "CLAIM-2026-00001",
            "incident_date": "2026-09-20",
            "location": "Bordeaux",
            "vehicle_make": "Renault",
            "vehicle_model": "Clio",
            "vehicle_year": 2021,
            "collision_type": "FRONT_COLLISION",
            "declared_damage": [
                "FRONT_BUMPER",
                "LEFT_HEADLIGHT",
            ],
            "injuries_declared": False,
        }
    )

    result = extract_claim_form_with_llm(
        extraction,
        llm,
    )

    assert result.claim_id == ("CLAIM-2026-00001")

    assert result.vehicle_make == "Renault"

    assert result.declared_damage == (
        DamageType.FRONT_BUMPER,
        DamageType.LEFT_HEADLIGHT,
    )


def test_extract_accident_report_with_llm():
    case = load_claim_case(Path("data/CLAIM-2026-00001"))

    assets = resolve_document_assets(case)

    asset = next(asset for asset in assets if asset.type == DocumentType.ACCIDENT_REPORT)

    extraction = extract_pdf_text(asset)

    llm = FakeStructuredLlm(
        {
            "claim_id": "CLAIM-2026-00001",
            "incident_date": "2026-09-20",
            "location": "Bordeaux",
            "vehicle_make": "Renault",
            "vehicle_model": "Clio",
            "collision_type": "FRONT_COLLISION",
            "reported_damage": [
                "FRONT_BUMPER",
                "LEFT_HEADLIGHT",
            ],
            "passengers": 1,
            "injuries_declared": False,
        }
    )

    result = extract_accident_report_with_llm(
        extraction,
        llm,
    )

    assert result.passengers == 1

    assert result.reported_damage == (
        DamageType.FRONT_BUMPER,
        DamageType.LEFT_HEADLIGHT,
    )
