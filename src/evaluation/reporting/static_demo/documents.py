"""Restyle controlled synthetic PDFs without recomputing recorded analyses."""

import shutil

from pypdf import PdfReader

from claims.assets import DocumentAsset
from claims.document_extraction import DocumentExtractionError
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from evaluation.deterministic.document_generation import STYLE_VERSION, render_page
from evaluation.deterministic.document_generation.stationery import document_kind
from evaluation.deterministic.documents import EXTRACTORS

LEGACY_STATIONERY = {
    "claim_form": (
        "ALIZE ASSURANCES",
        "MOTOR CLAIMS / POLICYHOLDER DECLARATION",
        "Please retain a copy of this declaration for your records.",
    ),
    "accident_report": (
        "ASTER EXPERTISE",
        "INCIDENT RECORD / CIRCUMSTANCES AND OBSERVATIONS",
        "Prepared for document review. Signature: ____________________",
    ),
    "garage_quote": (
        "ATELIER DES RIVES",
        "WORKSHOP ESTIMATE / SERVICE AND REPAIR DETAILS",
        "Work is subject to customer approval. No payment has been recorded.",
    ),
    "police_report": (
        "BUREAU DES DOSSIERS",
        "SUPPORTING RECORD / ADMINISTRATIVE COPY",
        "Supporting record only. Keep with the corresponding claim documents.",
    ),
}
LEGACY_FOOTER = (
    "Office hours: Monday to Friday, 09:00-18:00 | Correspondence by appointment",
    "Fictional organisation | Synthetic demonstration document | Reference copy",
)


def _business_text(text, name):
    """Recover only the exact legacy v1 envelope, never infer business values."""
    if text.lstrip().startswith("DEMO SYNTHETIC"):
        return text
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    company, caption, note = LEGACY_STATIONERY[name]
    if len(lines) < 9 or lines[0] != company or lines[4] != caption or lines[-3:] != (note, *LEGACY_FOOTER):
        raise ValueError("Only controlled synthetic demo PDFs can be restyled")
    return "\n".join((lines[3], *lines[5:-3]))


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
    name = document_kind(text, str(document_type).lower())
    expected = _structured(source, document_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    if reader.metadata and reader.metadata.subject == f"{STYLE_VERSION}:{name}":
        shutil.copyfile(source, target)
    else:
        body = _business_text(text, name)
        with pymupdf.open() as document:
            render_page(document, body, kind=name)
            document.save(target, no_new_id=True, garbage=4, deflate=True)
    if _structured(target, document_type) != expected:
        raise ValueError(f"Document restyling changed business information: {source.name}")
