from typing import TypeVar

from pydantic import BaseModel
from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.llm_document_extraction import (
    extract_garage_quote_with_llm,
)
from claims.loader import load_claim_case
import pytest

from claims.document_extraction import (
    DocumentExtractionError,
)

T = TypeVar(
    "T",
    bound=BaseModel,
)


class FakeStructuredLlm:
    def __init__(
        self,
        payload: dict,
    ):
        self.payload = payload

    def generate(
        self,
        *,
        prompt: str,
        response_model: type[T],
    ) -> T:
        return response_model.model_validate(
            self.payload
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
        DocumentExtractionError
    ):
        extract_garage_quote_with_llm(
            extraction,
            llm,
        )