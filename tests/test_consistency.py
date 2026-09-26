from pathlib import Path

from claims.assets import resolve_document_assets
from claims.consistency import compare_claim_documents
from claims.document_extraction import (
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
)
from claims.document_ingestion import extract_pdf_text
from claims.enums import DamageType, DocumentType
from claims.loader import load_claim_case

def test_compare_claim_documents():
    case = load_claim_case(
        Path("data/CLAIM-2026-00001")
    )

    assets = resolve_document_assets(case)

    def get_asset(document_type):
        return next(
            asset
            for asset in assets
            if asset.type == document_type
        )

    form = extract_claim_form(
        extract_pdf_text(
            get_asset(
                DocumentType.CLAIM_FORM
            )
        )
    )

    report = extract_accident_report(
        extract_pdf_text(
            get_asset(
                DocumentType.ACCIDENT_REPORT
            )
        )
    )

    quote = extract_garage_quote(
        extract_pdf_text(
            get_asset(
                DocumentType.REPAIR_QUOTE
            )
        )
    )
    consistency = compare_claim_documents(
        form,
        report,
        quote,
    )
    assert consistency.claim_id_match
    assert consistency.incident_date_match
    assert consistency.location_match
    assert consistency.vehicle_match
    assert consistency.collision_type_match
    assert consistency.injuries_match
    assert consistency.declared_damage_match
    assert consistency.additional_quote_damage == (
        DamageType.HOOD,
    )
    assert not consistency.is_consistent