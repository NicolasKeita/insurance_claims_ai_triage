"""A missing OCR dependency must not prevent reading native PDFs."""

import subprocess
import sys


def test_native_import_and_extraction_work_without_pytesseract():
    code = """
import importlib.abc
import sys
from pathlib import Path

class BlockPytesseract(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pytesseract" or fullname.startswith("pytesseract."):
            raise ModuleNotFoundError("pytesseract intentionally unavailable")

sys.meta_path.insert(0, BlockPytesseract())
from claims.assets import DocumentAsset
from claims.document_ingestion import extract_pdf_text, extract_pdf_text_auto
from claims.enums import DocumentType, TextExtractionMethod

assert "claims.document_ingestion.auto" not in sys.modules
asset = DocumentAsset(
    type=DocumentType.CLAIM_FORM,
    filename="claim_form.pdf",
    path=Path("data/CLAIM-2026-00001/documents/claim_form.pdf"),
)
assert extract_pdf_text(asset).has_text
result = extract_pdf_text_auto(asset, None)
assert result.has_text and result.method == TextExtractionMethod.NATIVE
assert "pytesseract" not in sys.modules
assert "claims.ocr" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
