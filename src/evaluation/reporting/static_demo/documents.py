"""Restyle controlled synthetic PDFs without recomputing recorded analyses."""

import shutil

from pypdf import PdfReader

from claims.assets import DocumentAsset
from claims.document_extraction import DocumentExtractionError
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from evaluation.deterministic.document_generation import render_page
from evaluation.deterministic.documents import EXTRACTORS


def _structured(path, kind):
    extractor = EXTRACTORS.get(kind)
    if extractor is None:
        return None
    asset = DocumentAsset(kind, path.name, path)
    try:
        return extractor(extract_pdf_text(asset)).model_dump(mode="json")
    except DocumentExtractionError as error:
        return {"extraction_error": str(error)}


def restyle_document(source, target, kind):
    """Keep business values, including intentionally invalid fixture fields."""
    import pymupdf

    reader = PdfReader(source)
    if reader.is_encrypted or len(reader.pages) != 1:
        raise ValueError("Static demo PDFs must be unencrypted single-page synthetic documents")
    text = reader.pages[0].extract_text() or ""
    document_type = DocumentType(kind)
    expected = _structured(source, document_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    if "Synthetic demonstration document" in text:
        shutil.copyfile(source, target)
    elif text.lstrip().startswith("DEMO SYNTHETIC"):
        with pymupdf.open() as document:
            render_page(document, text, kind=str(document_type).lower())
            document.save(target, no_new_id=True, garbage=4, deflate=True)
    else:
        raise ValueError("Only controlled synthetic demo PDFs can be restyled")
    if _structured(target, document_type) != expected:
        raise ValueError(f"Document restyling changed business information: {source.name}")
