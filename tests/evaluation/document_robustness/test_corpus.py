"""Ground truth remains independent of transformed PDFs and production parsers."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from claims.assets import DocumentAsset
from claims.consistency import compare_claim_documents
from claims.document_extraction import extract_accident_report, extract_claim_form, extract_garage_quote
from claims.document_ingestion import extract_pdf_text
from claims.document_models import MoneyAmount
from evaluation.deterministic.document_robustness.corpus import TYPES, build_corpus
from evaluation.deterministic.document_robustness.transformations import materialize_variant

EXTRACTORS = (extract_claim_form, extract_accident_report, extract_garage_quote)


def test_corpus_source_ground_truth_and_variation():
    corpus = build_corpus()
    assert len(corpus) == 5
    assert len({item.scenario_id for item in corpus}) == 5
    assert {document.document_type for item in corpus for document in item.documents} == {
        "claim_form",
        "accident_report",
        "garage_quote",
    }
    assert len({item.documents[0].expected.claim_id for item in corpus}) == 5
    assert len({item.documents[0].expected.incident_date for item in corpus}) == 5
    assert {item.documents[2].expected.total.currency for item in corpus} == {"EUR", "USD", "GBP"}
    for item in corpus:
        form, report, quote = (document.expected for document in item.documents)
        assert compare_claim_documents(form, report, quote).model_dump(mode="json") == item.expected_consistency
        assert quote.total.amount == sum((part.price.amount for part in quote.parts), quote.labor.amount)


def test_native_extraction_matches_source_models(tmp_path):
    for scenario in build_corpus():
        for document, kind, extractor in zip(scenario.documents, TYPES, EXTRACTORS, strict=True):
            path = tmp_path / scenario.scenario_id / f"{document.document_type}.pdf"
            materialize_variant(document, "native_clean", path)
            asset = DocumentAsset(kind, path.name, path)
            actual = extractor(extract_pdf_text(asset))
            assert actual.model_dump() == document.expected.model_dump()


def test_ground_truth_financial_schema_rejects_invalid_currency_and_amount():
    with pytest.raises(ValidationError):
        MoneyAmount(amount=Decimal("-1"), currency="EUR")
    with pytest.raises(ValidationError):
        MoneyAmount(amount=Decimal("1200"), currency="EURO")


def test_expected_mismatch_is_defined_before_pdf_generation():
    corpus = build_corpus()
    assert corpus[2].expected_consistency["additional_quote_damage"] == ["HOOD"]
    assert corpus[3].expected_consistency["incident_date_match"] is False
    assert corpus[0].expected_consistency["incident_date_match"] is True
