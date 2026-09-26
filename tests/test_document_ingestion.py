import shutil

import pytest

from claims.document_ingestion import (
    DocumentIngestionError,
    extract_pdf_text,
)

from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case

def test_extract_repair_quote_text():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    repair_quote = next(
        asset
        for asset in assets
        if asset.type == DocumentType.REPAIR_QUOTE
    )

    extraction = extract_pdf_text(
        repair_quote
    )

    assert extraction.page_count == 1
    assert extraction.has_text

    assert "GARAGE QUOTE" in extraction.text
    assert "GARAGE-42" in extraction.text
    assert "3160 EUR" in extraction.text

def test_extract_claim_form_text():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    claim_form = next(
        asset
        for asset in assets
        if asset.type == DocumentType.CLAIM_FORM
    )

    extraction = extract_pdf_text(
        claim_form
    )

    assert extraction.has_text

    assert (
        "CLAIM-2026-00001"
        in extraction.text
    )

    assert "Renault Clio" in extraction.text
    assert "Bordeaux" in extraction.text

def test_fake_pdf_is_rejected(tmp_path):
    source_dir = Path(
        "data/CLAIM-2026-00001"
    )

    claim_dir = (
        tmp_path / "CLAIM-2026-00001"
    )

    shutil.copytree(
        source_dir,
        claim_dir,
    )

    fake_pdf = (
        claim_dir
        / "documents"
        / "garage_quote.pdf"
    )

    fake_pdf.write_text(
        "This is not a real PDF.",
        encoding="utf-8",
    )

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    repair_quote = next(
        asset
        for asset in assets
        if asset.type == DocumentType.REPAIR_QUOTE
    )

    with pytest.raises(
        DocumentIngestionError
    ):
        extract_pdf_text(repair_quote)


def test_all_available_documents_can_be_read():
    claim_dir = Path(
        "data/CLAIM-2026-00001"
    )

    case = load_claim_case(claim_dir)

    assets = resolve_document_assets(case)

    assert len(assets) == 3

    for asset in assets:
        extraction = extract_pdf_text(asset)

        assert extraction.page_count >= 1
        assert extraction.has_text