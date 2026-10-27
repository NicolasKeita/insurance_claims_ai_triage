"""Reject ambiguous financial text instead of inventing OCR corrections."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from claims.assets import DocumentAsset, resolve_document_assets
from claims.consistency import compare_claim_documents
from claims.document_extraction import (
    DocumentExtractionError,
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
    find_prefixed_value,
    parse_document_date,
)
from claims.document_ingestion import PdfTextExtraction, extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case


def quote_text(part="1200.00", total="1200.00", extra=""):
    asset = DocumentAsset(type=DocumentType.REPAIR_QUOTE, filename="demo.pdf", path=Path("demo.pdf"))
    return PdfTextExtraction(
        asset=asset,
        pages=(f"Garage: DEMO\nClaim ID: DEMO-1\nFront bumper: {part} EUR\nLabor: 0 EUR\n{extra}Total: {total} EUR",),
    )


@pytest.mark.parametrize("value", ["0", "1", "1200.00"])
def test_unambiguous_money_keeps_decimal_value(value):
    result = extract_garage_quote(quote_text(part=value, total=value))
    assert result.parts[0].price.amount == Decimal(value)
    assert result.total.amount == Decimal(value)


@pytest.mark.parametrize("value", ["1 200.00", "3,160", "O", "I", "12O0.00", "12I0.00"])
def test_ambiguous_money_is_not_corrected(value):
    with pytest.raises(DocumentExtractionError, match="Invalid quote amount"):
        extract_garage_quote(quote_text(part=value, total=value))


@pytest.mark.parametrize("value", ["1 200.00", "3,160", "O", "I"])
def test_ambiguous_total_is_not_corrected(value):
    with pytest.raises(DocumentExtractionError, match="Invalid quote amount"):
        extract_garage_quote(quote_text(total=value))


def test_malformed_zero_price_line_cannot_silently_disappear():
    with pytest.raises(DocumentExtractionError, match="Invalid quote amount"):
        extract_garage_quote(quote_text(extra="Hood: O EUR\n"))


def test_conflicting_totals_cannot_silently_overwrite_each_other():
    with pytest.raises(DocumentExtractionError, match="Conflicting total"):
        extract_garage_quote(quote_text(extra="Total: 9999 EUR\n"))


@pytest.mark.parametrize("prefix", ["Claim ID:", "Incident date:"])
def test_conflicting_critical_fields_fail_closed(prefix):
    with pytest.raises(DocumentExtractionError, match="Conflicting values"):
        find_prefixed_value([prefix + " first", prefix + " second"], prefix)
    assert find_prefixed_value([prefix + " same", prefix + " same"], prefix) == "same"


@pytest.mark.parametrize("value", ["2O/09/2026", "20/09/2O26", "31/09/2026", "2026-09-20"])
def test_invalid_dates_are_not_repaired(value):
    with pytest.raises(DocumentExtractionError, match="Invalid document date"):
        parse_document_date(value)


@pytest.mark.parametrize("value, expected", [("20/09/2026", date(2026, 9, 20)), ("29/02/2024", date(2024, 2, 29))])
def test_valid_document_dates_keep_exact_value(value, expected):
    assert parse_document_date(value) == expected


def test_valid_but_changed_date_is_detected_downstream():
    case = load_claim_case(Path("data/CLAIM-2026-00001"))
    assets = {asset.type: asset for asset in resolve_document_assets(case)}
    form = extract_claim_form(extract_pdf_text(assets[DocumentType.CLAIM_FORM]))
    raw_report = extract_pdf_text(assets[DocumentType.ACCIDENT_REPORT])
    changed_report = PdfTextExtraction(
        asset=raw_report.asset,
        pages=tuple(page.replace("20/09/2026", "28/09/2026") for page in raw_report.pages),
    )
    report = extract_accident_report(changed_report)
    quote = extract_garage_quote(extract_pdf_text(assets[DocumentType.REPAIR_QUOTE]))
    assert report.incident_date == date(2026, 9, 28)
    assert not compare_claim_documents(form, report, quote).incident_date_match
