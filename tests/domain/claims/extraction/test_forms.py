"""Test forms."""

from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_extraction import (
    extract_accident_report,
    extract_claim_form,
)
from claims.document_ingestion import (
    extract_pdf_text,
)
from claims.enums import DamageType, DocumentType
from claims.loader import load_claim_case


def test_extract_structured_claim_form():
    claim_dir = Path("data/CLAIM-2026-00001")

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    asset = next(asset for asset in assets if asset.type == DocumentType.CLAIM_FORM)

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

    asset = next(asset for asset in assets if asset.type == DocumentType.ACCIDENT_REPORT)

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
