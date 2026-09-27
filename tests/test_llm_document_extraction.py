from decimal import Decimal
from typing import TypeVar

from pydantic import BaseModel
from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.llm_document_extraction import (
    extract_accident_report_with_llm,
    extract_claim_form_with_llm,
    extract_document_with_llm,
    extract_garage_quote_with_llm,
)
from claims.loader import load_claim_case
import pytest

from claims.document_extraction import (
    DocumentExtractionError,
)
from claims.document_models import ClaimFormExtraction
from claims.enums import DamageType

T = TypeVar(
    "T",
    bound=BaseModel,
)


class FakeStructuredLlm:
    def __init__(
        self,
        payloads: dict | list[dict],
    ):
        if isinstance(payloads, dict):
            payloads = [payloads]
        self.payloads = payloads
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        prompt: str,
        response_model: type[T],
    ) -> T:
        index = min(
            self.call_count,
            len(self.payloads) - 1,
        )
        self.call_count += 1
        self.prompts.append(prompt)
        return response_model.model_validate(
            self.payloads[index]
        )

def test_extract_garage_quote_with_llm():
    case = load_claim_case(
        Path("data/CLAIM-2026-00001")
    )

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.REPAIR_QUOTE
    )

    extraction = extract_pdf_text(asset)

    llm = FakeStructuredLlm(
        {
            "garage": "GARAGE-42",
            "claim_id": "CLAIM-2026-00001",
            "parts": [
                {
                    "description": "Front bumper",
                    "price": {
                        "amount": "780",
                        "currency": "EUR",
                    },
                },
                {
                    "description": "Left headlight",
                    "price": {
                        "amount": "620",
                        "currency": "EUR",
                    },
                },
                {
                    "description": "Hood",
                    "price": {
                        "amount": "840",
                        "currency": "EUR",
                    },
                },
            ],
            "labor": {
                "amount": "920",
                "currency": "EUR",
            },
            "total": {
                "amount": "3160",
                "currency": "EUR",
            },
        }
    )

    quote = extract_garage_quote_with_llm(
        extraction,
        llm,
    )

    assert quote.garage == "GARAGE-42"
    assert quote.claim_id == "CLAIM-2026-00001"
    assert len(quote.parts) == 3
    assert quote.source.filename == (
        "garage_quote.pdf"
    )

    assert (
        quote.source.document_type
        == DocumentType.REPAIR_QUOTE
    )
    assert llm.call_count == 1
    llm = FakeStructuredLlm(
        {
            "garage": None,
            "claim_id": "CLAIM-2026-00001",
            "parts": [],
            "labor": {
                "amount": "920",
                "currency": "EUR",
            },
            "total": {
                "amount": "920",
                "currency": "EUR",
            },
        }
    )

    with pytest.raises(
        DocumentExtractionError,
        match="Missing required quote fields after retry: garage",
    ):
        extract_garage_quote_with_llm(
            extraction,
            llm,
        )
    assert llm.call_count == 2

def test_garage_quote_retries_missing_fields():
    case = load_claim_case(Path("data/CLAIM-2026-00001"))
    asset = next(
        asset
        for asset in resolve_document_assets(case)
        if asset.type == DocumentType.REPAIR_QUOTE
    )
    extraction = extract_pdf_text(asset)

    parts = [
        {
            "description": description,
            "price": {"amount": amount, "currency": "EUR"},
        }
        for description, amount in (
            ("Front bumper", 780),
            ("Left headlight", 620),
            ("Hood", 840),
        )
    ]
    llm = FakeStructuredLlm([
        {
            "garage": "GARAGE-42",
            "claim_id": "CLAIM-2026-00001",
            "parts": parts,
            "labor": None,
            "total": None,
        },
        {
            "garage": "GARAGE-42",
            "claim_id": "CLAIM-2026-00001",
            "parts": parts,
            "labor": {"amount": 920, "currency": "EUR"},
            "total": {"amount": 3160, "currency": "EUR"},
        },
    ])

    quote = extract_garage_quote_with_llm(extraction, llm)

    assert llm.call_count == 2
    assert "labor, total" in llm.prompts[1]
    assert "Labor: 920 EUR" in llm.prompts[1]
    assert quote.labor.amount == Decimal("920")
    assert quote.total.amount == Decimal("3160")

def test_extract_claim_form_with_llm():
    case = load_claim_case(
        Path("data/CLAIM-2026-00001")
    )

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.CLAIM_FORM
    )

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

    assert result.claim_id == (
        "CLAIM-2026-00001"
    )

    assert result.vehicle_make == "Renault"

    assert result.declared_damage == (
        DamageType.FRONT_BUMPER,
        DamageType.LEFT_HEADLIGHT,
    )

def test_extract_accident_report_with_llm():
    case = load_claim_case(
        Path("data/CLAIM-2026-00001")
    )

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.ACCIDENT_REPORT
    )

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

def test_llm_document_dispatcher():
    case = load_claim_case(
        Path("data/CLAIM-2026-00001")
    )

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.CLAIM_FORM
    )

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
