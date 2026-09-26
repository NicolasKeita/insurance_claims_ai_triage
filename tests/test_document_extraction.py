from decimal import Decimal
from pathlib import Path

import pytest

from claims.assets import resolve_document_assets
from claims.document_extraction import (
    DocumentExtractionError,
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
)
from claims.document_ingestion import (
    PdfTextExtraction,
    extract_pdf_text,
)
from claims.enums import DamageType, DocumentType
from claims.loader import load_claim_case


def test_extract_structured_garage_quote():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.REPAIR_QUOTE
    )

    text_extraction = extract_pdf_text(asset)

    quote = extract_garage_quote(text_extraction)

    assert quote.garage == "GARAGE-42"
    assert quote.claim_id == "CLAIM-2026-00001"

    assert quote.total.amount == Decimal("3160")
    assert quote.total.currency == "EUR"

    assert len(quote.parts) == 3

    assert quote.parts[0].description == "Front bumper"
    assert quote.parts[0].price.amount == Decimal("780")

    hood = next(
        item
        for item in quote.parts
        if item.description == "Hood"
    )

    assert hood.price.amount == Decimal("840")

    assert (
        quote.source.document_type
        == DocumentType.REPAIR_QUOTE
    )
    assert quote.source.filename == "garage_quote.pdf"
    assert quote.source.page == 1


def test_inconsistent_quote_total_is_rejected():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.REPAIR_QUOTE
    )

    bad_text = """
GARAGE QUOTE

Garage: GARAGE-42
Claim ID: CLAIM-2026-00001

Front bumper: 780 EUR
Left headlight: 620 EUR
Hood: 840 EUR
Labor: 920 EUR

Total: 9999 EUR
"""

    extraction = PdfTextExtraction(
        asset=asset,
        pages=(bad_text,),
    )

    with pytest.raises(DocumentExtractionError):
        extract_garage_quote(extraction)


def test_extract_structured_claim_form():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.CLAIM_FORM
    )

    text_extraction = extract_pdf_text(asset)

    form = extract_claim_form(text_extraction)

    assert form.claim_id == "CLAIM-2026-00001"
    assert form.location == "Bordeaux"

    assert form.vehicle_make == "Renault"
    assert form.vehicle_model == "Clio"
    assert form.vehicle_year == 2021

    assert not form.injuries_declared

    assert form.declared_damage == (
        DamageType.FRONT_BUMPER,
        DamageType.LEFT_HEADLIGHT,
    )


def test_extract_structured_accident_report():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    asset = next(
        asset
        for asset in assets
        if asset.type == DocumentType.ACCIDENT_REPORT
    )

    text_extraction = extract_pdf_text(asset)

    report = extract_accident_report(text_extraction)

    assert report.claim_id == "CLAIM-2026-00001"
    assert report.location == "Bordeaux"

    assert report.vehicle_make == "Renault"
    assert report.vehicle_model == "Clio"

    assert report.passengers == 1
    assert not report.injuries_declared

    assert report.reported_damage == (
        DamageType.FRONT_BUMPER,
        DamageType.LEFT_HEADLIGHT,
    )