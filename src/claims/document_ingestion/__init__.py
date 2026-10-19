"""PDF ingestion: native text first, OCR only for pages without text."""

from claims.document_ingestion.auto import extract_pdf_text_auto
from claims.document_ingestion.pdf import DocumentIngestionError, PdfTextExtraction, extract_pdf_text

__all__ = [
    "DocumentIngestionError",
    "PdfTextExtraction",
    "extract_pdf_text",
    "extract_pdf_text_auto",
]
