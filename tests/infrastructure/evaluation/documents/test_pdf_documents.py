"""Professional document decoration preserves the original business evidence."""

from pathlib import Path

import pymupdf
import pytest

from claims.assets import DocumentAsset
from claims.consistency import compare_claim_documents
from claims.document_extraction import (
    DocumentExtractionError,
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
)
from claims.document_ingestion import PdfTextExtraction, extract_pdf_text
from claims.enums import DamageType, DocumentType
from evaluation.deterministic.document_generation import render_page, visible_text
from tests.infrastructure.evaluation.documents._pdf_documents import FORM, QUOTE, REPORT

DOCUMENTS = (
    ("claim_form", DocumentType.CLAIM_FORM, FORM, extract_claim_form),
    ("accident_report", DocumentType.ACCIDENT_REPORT, REPORT, extract_accident_report),
    ("garage_quote", DocumentType.REPAIR_QUOTE, QUOTE, extract_garage_quote),
)


def rendered(tmp_path, kind, doc_type, body):
    path = tmp_path / f"{kind}.pdf"
    with pymupdf.open() as document:
        render_page(document, body, kind=kind)
        document.save(path, no_new_id=True, garbage=4, deflate=True)
    asset = DocumentAsset(doc_type, path.name, path)
    return extract_pdf_text(asset)


@pytest.mark.parametrize("kind,doc_type,body,extractor", DOCUMENTS, ids=("claim", "report", "quote"))
def test_branding_and_administrative_text_preserve_business_model(tmp_path, kind, doc_type, body, extractor):
    decorated = rendered(tmp_path, kind, doc_type, body)
    plain = PdfTextExtraction(asset=decorated.asset, pages=(body,))
    assert extractor(decorated).model_dump() == extractor(plain).model_dump()
    assert decorated.page_count == 1
    assert " ".join(decorated.text.split()) == " ".join(visible_text(body, kind=kind).split())
    assert len(decorated.text.split()) > len(body.split())


@pytest.mark.parametrize("kind,doc_type,body,extractor", DOCUMENTS, ids=("claim", "report", "quote"))
def test_checked_in_documents_keep_original_business_evidence(kind, doc_type, body, extractor):
    path = Path("data/CLAIM-2026-00001/documents") / f"{kind}.pdf"
    asset = DocumentAsset(doc_type, path.name, path)
    original = PdfTextExtraction(asset=asset, pages=(body,))
    assert extractor(extract_pdf_text(asset)).model_dump() == extractor(original).model_dump()


def test_document_set_keeps_the_deliberate_additional_hood_damage(tmp_path):
    values = [extractor(rendered(tmp_path, kind, doc_type, body)) for kind, doc_type, body, extractor in DOCUMENTS]
    form, report, quote = values
    consistency = compare_claim_documents(form, report, quote)
    assert consistency.claim_id_match and consistency.incident_date_match
    assert consistency.declared_damage_match and consistency.vehicle_match
    assert consistency.additional_quote_damage == (DamageType.HOOD,)
    assert str(quote.total.amount) == "3160"
    assert quote.garage == "GARAGE-42"


@pytest.mark.parametrize("kind,doc_type,body,extractor", DOCUMENTS, ids=("claim", "report", "quote"))
def test_regeneration_is_byte_identical_without_pdf_metadata_dates(tmp_path, kind, doc_type, body, extractor):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = rendered(left, kind, doc_type, body)
    second = rendered(right, kind, doc_type, body)
    assert first.asset.path.read_bytes() == second.asset.path.read_bytes()


@pytest.mark.parametrize(
    "extra,error",
    [("\nTotal: 9999 EUR", "Conflicting total"), ("\nClaim ID: OTHER-CLAIM", "Conflicting values")],
)
def test_decoration_does_not_hide_conflicting_business_evidence(tmp_path, extra, error):
    extraction = rendered(tmp_path, "garage_quote", DocumentType.REPAIR_QUOTE, QUOTE + extra)
    with pytest.raises(DocumentExtractionError, match=error):
        extract_garage_quote(extraction)


def test_decoration_does_not_correct_an_invalid_financial_total(tmp_path):
    body = QUOTE.replace("Total: 3160 EUR", "Total: 9999 EUR")
    extraction = rendered(tmp_path, "garage_quote", DocumentType.REPAIR_QUOTE, body)
    with pytest.raises(DocumentExtractionError, match="total does not match"):
        extract_garage_quote(extraction)


@pytest.mark.parametrize("body", ["Line of administrative text\n" * 100, "W" * 1000])
def test_content_that_exceeds_one_page_is_rejected_instead_of_truncated(body):
    with pymupdf.open() as document, pytest.raises(ValueError):
        render_page(document, body, kind="claim_form")
