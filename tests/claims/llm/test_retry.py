"""Test retry."""

from decimal import Decimal

from claims.llm_document_extraction import (
    extract_garage_quote_with_llm,
)
from tests.claims.llm._llm import FakeStructuredLlm
from tests.claims.llm._quote import quote_extraction


def test_garage_quote_retries_missing_fields():
    extraction, parts = quote_extraction()
    llm = FakeStructuredLlm(
        [
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
        ]
    )

    quote = extract_garage_quote_with_llm(extraction, llm)

    assert llm.call_count == 2
    assert "labor, total" in llm.prompts[1]
    assert "Labor: 920 EUR" in llm.prompts[1]
    assert quote.labor.amount == Decimal("920")
    assert quote.total.amount == Decimal("3160")
