"""Test dispatch."""

from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.document_models import ClaimFormExtraction
from claims.enums import DocumentType
from claims.llm_document_extraction import (
    extract_document_with_llm,
)
from claims.loader import load_claim_case
from tests.claims.llm._llm import FakeStructuredLlm


def test_llm_document_dispatcher():
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

    result = extract_document_with_llm(
        extraction,
        llm,
    )

    assert isinstance(
        result,
        ClaimFormExtraction,
    )
