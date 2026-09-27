from pathlib import Path

from claims.assets import DocumentAsset
from claims.document_ingestion import (
    extract_pdf_text_auto,
)
from claims.document_extraction import (
    extract_accident_report,
)
from claims.enums import (
    DocumentType,
    TextExtractionMethod,
)

class FakeOcrEngine:
    def __init__(
        self,
        text: str,
    ):
        self.text = text
        self.calls = 0

    def extract_text(
        self,
        image,
    ) -> str:
        self.calls += 1
        return self.text

def test_native_pdf_does_not_use_ocr():
    asset = DocumentAsset(
        type=DocumentType.ACCIDENT_REPORT,
        filename="accident_report.pdf",
        path=Path(
            "data/CLAIM-2026-00001/"
            "documents/accident_report.pdf"
        ),
    )

    ocr = FakeOcrEngine(
        "THIS SHOULD NEVER BE USED"
    )

    extraction = extract_pdf_text_auto(
        asset,
        ocr,
    )

    assert (
        extraction.method
        == TextExtractionMethod.NATIVE
    )

    assert ocr.calls == 0

    assert "ACCIDENT REPORT" in (
        extraction.text
    )

SCAN_TEXT = """
ACCIDENT REPORT

Claim ID: CLAIM-2026-00001

Date: 20/09/2026
Location: Bordeaux

Collision type: Front collision

Vehicle:
Renault Clio

Reported damage:
Front bumper
Left headlight

Passengers: 1
Injuries: No
""".strip()

def test_scanned_pdf_uses_ocr():
    asset = DocumentAsset(
        type=DocumentType.ACCIDENT_REPORT,
        filename="accident_report_scan.pdf",
        path=Path(
            "tests/data/ocr/"
            "accident_report_scan.pdf"
        ),
    )

    ocr = FakeOcrEngine(
        SCAN_TEXT
    )

    extraction = extract_pdf_text_auto(
        asset,
        ocr,
    )

    assert (
        extraction.method
        == TextExtractionMethod.OCR
    )

    assert ocr.calls == 1

    assert "ACCIDENT REPORT" in (
        extraction.text
    )

    report = extract_accident_report(
        extraction
    )

    assert (
        report.claim_id
        == "CLAIM-2026-00001"
    )

    assert report.location == "Bordeaux"
    assert report.passengers == 1