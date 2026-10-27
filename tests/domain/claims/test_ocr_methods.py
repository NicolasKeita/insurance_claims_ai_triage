"""Method labels and timings describe the pages actually processed."""

import pymupdf

from claims.assets import DocumentAsset
from claims.document_ingestion import extract_pdf_text, extract_pdf_text_auto
from claims.enums import DocumentType, TextExtractionMethod
from tests.domain.claims.test_ocr_ingestion import FakeOcrEngine


def image_only_asset(tmp_path, *, native_page=False):
    path = tmp_path / "demo.pdf"
    with pymupdf.open() as document:
        if native_page:
            document.new_page().insert_text((36, 36), "Native page")
        page = document.new_page(width=72, height=72)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 72, 72), False)
        pixmap.clear_with(255)
        page.insert_image(page.rect, pixmap=pixmap)
        document.save(path)
    return DocumentAsset(type=DocumentType.CLAIM_FORM, filename=path.name, path=path)


def test_image_only_page_has_no_native_text_and_remains_ocr_when_empty(tmp_path):
    asset = image_only_asset(tmp_path)
    assert not extract_pdf_text(asset).has_text
    engine = FakeOcrEngine("")
    result = extract_pdf_text_auto(asset, engine, dpi=72)
    assert result.method == TextExtractionMethod.OCR
    assert not result.has_text
    assert engine.calls == 1
    assert result.native_duration_seconds > 0
    assert result.ocr_duration_seconds > 0


def test_mixed_pages_report_hybrid_and_only_ocr_empty_page(tmp_path):
    engine = FakeOcrEngine("Scanned page")
    result = extract_pdf_text_auto(image_only_asset(tmp_path, native_page=True), engine, dpi=72)
    assert result.method == TextExtractionMethod.HYBRID
    assert result.pages == ("Native page", "Scanned page")
    assert engine.calls == 1
    assert result.native_duration_seconds > 0
    assert result.ocr_duration_seconds > 0


def test_native_page_has_native_timing_and_zero_ocr_timing(tmp_path):
    path = tmp_path / "native.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((36, 36), "Native page")
        document.save(path)
    asset = DocumentAsset(type=DocumentType.CLAIM_FORM, filename=path.name, path=path)
    engine = FakeOcrEngine("unused")
    result = extract_pdf_text_auto(asset, engine)
    assert result.method == TextExtractionMethod.NATIVE
    assert result.native_duration_seconds > 0
    assert result.ocr_duration_seconds == 0
    assert engine.calls == 0
