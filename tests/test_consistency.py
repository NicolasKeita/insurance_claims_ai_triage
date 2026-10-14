from pathlib import Path

from claims.assets import resolve_document_assets
from claims.consistency import compare_claim_documents, map_quote_damage_types
from claims.document_models import AccidentReportExtraction, ClaimFormExtraction, GarageQuoteExtraction
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


def make_quote(*descriptions):
    return GarageQuoteExtraction(
        garage="Fixture garage", claim_id="fixture",
        parts=tuple({"description": text, "price": {"amount": 10, "currency": "EUR"}}
                    for text in descriptions),
        labor={"amount": 0, "currency": "EUR"},
        total={"amount": len(descriptions) * 10, "currency": "EUR"},
        source={"document_type": "REPAIR_QUOTE", "filename": "fixture.pdf", "page": 1},
    )


def test_central_quote_mapping_preserves_exact_current_semantics_and_occurrences():
    mapped, unmapped = map_quote_damage_types(make_quote("HOOD", "hood", "Front Bumper", "paint", "paint"))
    assert mapped == (DamageType.HOOD, DamageType.HOOD, DamageType.FRONT_BUMPER)
    assert unmapped == ("paint", "paint")
    assert map_quote_damage_types(make_quote()) == ((), ())


def test_document_comparison_handles_quote_with_only_unmapped_lines():
    common = {
        "claim_id": "fixture", "incident_date": "2026-01-01", "location": "Paris",
        "vehicle_make": "Fixture", "vehicle_model": "Car", "collision_type": "FRONT_COLLISION",
        "injuries_declared": False,
    }
    form = ClaimFormExtraction(
        **common, vehicle_year=2020, declared_damage=(DamageType.HOOD,),
        source={"document_type": "CLAIM_FORM", "filename": "form.pdf", "page": 1},
    )
    report = AccidentReportExtraction(
        **common, reported_damage=(DamageType.HOOD,), passengers=0,
        source={"document_type": "ACCIDENT_REPORT", "filename": "report.pdf", "page": 1},
    )
    result = compare_claim_documents(form, report, make_quote("paint labor"))
    assert result.additional_quote_damage == ()
    assert result.unmapped_quote_items == ("paint labor",)
