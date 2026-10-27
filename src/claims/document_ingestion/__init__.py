"""PDF ingestion: native text first, OCR only for pages without text."""

from claims.document_ingestion.pdf import DocumentIngestionError, PdfTextExtraction, extract_pdf_text

__all__ = [
    "DocumentIngestionError",
    "PdfTextExtraction",
    "extract_pdf_text",
    "extract_pdf_text_auto",
]


def extract_pdf_text_auto(asset, ocr_engine, dpi: int = 300) -> PdfTextExtraction:
    """Load raster ingestion only when requested; native imports stay independent."""
    from claims.document_ingestion.auto import extract_pdf_text_auto as extract_auto

    return extract_auto(asset, ocr_engine, dpi=dpi)
