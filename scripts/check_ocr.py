from pathlib import Path

from claims.assets import DocumentAsset
from claims.document_extraction import (
    extract_accident_report,
)
from claims.document_ingestion import (
    extract_pdf_text_auto,
)
from claims.enums import DocumentType
from claims.ocr import TesseractOcrEngine


asset = DocumentAsset(
    type=DocumentType.ACCIDENT_REPORT,
    filename="accident_report_scan.pdf",
    path=Path(
        "tests/data/ocr/"
        "accident_report_scan.pdf"
    ),
)


ocr = TesseractOcrEngine(
    language="eng"
)


extraction = extract_pdf_text_auto(
    asset,
    ocr,
)


print(
    f"Extraction method: "
    f"{extraction.method}"
)

print()
print("=" * 60)
print("OCR TEXT")
print("=" * 60)

print(extraction.text)


report = extract_accident_report(
    extraction
)


print()
print("=" * 60)
print("STRUCTURED RESULT")
print("=" * 60)

print(
    report.model_dump_json(
        indent=2
    )
)