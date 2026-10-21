"""Test quote."""

from pathlib import Path

import pytest

from claims.assets import resolve_document_assets
from claims.document_extraction import (
    DocumentExtractionError,
)
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.llm_document_extraction import (
    extract_garage_quote_with_llm,
)
from claims.loader import load_claim_case
from tests.claims.llm._llm import FakeStructuredLlm
from tests.claims.llm._quote import quote_llm


def test_extract_garage_quote_with_llm():
    case = load_claim_case(Path("data/CLAIM-2026-00001"))

    assets = resolve_document_assets(case)

    asset = next(asset for asset in assets if asset.type == DocumentType.REPAIR_QUOTE)

    extraction = extract_pdf_text(asset)

    llm = quote_llm()
    quote = extract_garage_quote_with_llm(
        extraction,
        llm,
    )

    assert quote.garage == "GARAGE-42"
    assert quote.claim_id == "CLAIM-2026-00001"
    assert len(quote.parts) == 3
    assert quote.source.filename == ("garage_quote.pdf")

    assert quote.source.document_type == DocumentType.REPAIR_QUOTE
    assert llm.call_count == 1
    assert_missing_garage(extraction)


def assert_missing_garage(extraction):
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
