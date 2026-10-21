"""Quote."""

from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case
from tests.claims.llm._llm import FakeStructuredLlm


def quote_llm():
    return FakeStructuredLlm(
        {
            "garage": "GARAGE-42",
            "claim_id": "CLAIM-2026-00001",
            "parts": quote_parts(),
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


def quote_parts():
    return [
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
    ]


def quote_extraction():
    case = load_claim_case(Path("data/CLAIM-2026-00001"))
    asset = next(asset for asset in resolve_document_assets(case) if asset.type == DocumentType.REPAIR_QUOTE)
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
    return extraction, parts
