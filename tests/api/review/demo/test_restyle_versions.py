"""Older branded documents migrate without duplicate stationery or lost facts."""

import pymupdf
import pytest
from pypdf import PdfReader

from claims.assets import DocumentAsset
from claims.document_extraction import extract_claim_form
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from evaluation.deterministic.document_generation import STYLE_VERSION
from evaluation.reporting.static_demo.documents import restyle_document
from tests.api.review.demo._static_demo import BODY

LEGACY_HEADER = (
    "ALIZE ASSURANCES\nMotor insurance | Policyholder services\n"
    "18 Avenue des Amandiers, Bordeaux | contact@alize.example\n"
)
LEGACY_CAPTION = "MOTOR CLAIMS / POLICYHOLDER DECLARATION"
LEGACY_TAIL = (
    "Please retain a copy of this declaration for your records.\n"
    "Office hours: Monday to Friday, 09:00-18:00 | Correspondence by appointment\n"
    "Fictional organisation | Synthetic demonstration document | Reference copy"
)


def legacy_pdf(path, subject, *, company="ALIZE ASSURANCES"):
    title, body = BODY.split("\n", 1)
    text = f"{LEGACY_HEADER}{title}\n{LEGACY_CAPTION}\n{body}\n{LEGACY_TAIL}"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((45, 45), text.replace("ALIZE ASSURANCES", company), fontsize=10)
        if subject:
            pdf.set_metadata({"subject": subject})
        pdf.save(path, no_new_id=True)


@pytest.mark.parametrize("subject", [None, "professional_stationery_v1:claim_form"])
def test_legacy_envelope_is_replaced_and_source_business_facts_are_preserved(tmp_path, subject):
    source, target = tmp_path / "legacy.pdf", tmp_path / "current" / "legacy.pdf"
    legacy_pdf(source, subject)
    original_bytes = source.read_bytes()
    before = DocumentAsset(DocumentType.CLAIM_FORM, source.name, source)
    after = DocumentAsset(DocumentType.CLAIM_FORM, source.name, target)
    restyle_document(source, target, "CLAIM_FORM")
    assert extract_claim_form(extract_pdf_text(after)) == extract_claim_form(extract_pdf_text(before))
    assert source.read_bytes() == original_bytes
    assert target.read_bytes() != original_bytes
    reader = PdfReader(target)
    assert reader.metadata.subject == f"{STYLE_VERSION}:claim_form"
    assert reader.pages[0].extract_text().count("Claim ID:") == 1
    copy = tmp_path / "copy" / "legacy.pdf"
    restyle_document(target, copy, "CLAIM_FORM")
    assert copy.read_bytes() == target.read_bytes()


def test_a_synthetic_footer_without_the_known_legacy_envelope_is_not_trusted(tmp_path):
    source, target = tmp_path / "unrelated.pdf", tmp_path / "current.pdf"
    legacy_pdf(source, None, company="UNRELATED OFFICE")
    with pytest.raises(ValueError, match="controlled synthetic"):
        restyle_document(source, target, "CLAIM_FORM")
    assert not target.exists()
