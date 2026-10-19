"""Automatic native-then-OCR PDF text extraction."""

import pymupdf
from PIL import Image

from claims.assets import DocumentAsset
from claims.document_ingestion.pdf import DocumentIngestionError, PdfTextExtraction, extract_pdf_text
from claims.enums import TextExtractionMethod
from claims.ocr import OcrEngine


def extract_pdf_text_auto(asset: DocumentAsset, ocr_engine: OcrEngine, dpi: int = 300) -> PdfTextExtraction:
    """
    Every page already contains usable native text: OCR is unnecessary.

    Keep native text whenever available.

    No native text: render this page and use OCR.
    """
    native = extract_pdf_text(asset)
    if all(page.strip() for page in native.pages):
        return native
    document = _open_document(asset)
    try:
        pages, methods = _render_pages(document, native.pages, ocr_engine, dpi)
    finally:
        document.close()
    return PdfTextExtraction(asset=asset, pages=tuple(pages), method=_extraction_method(methods))


def _open_document(asset):
    try:
        return pymupdf.open(str(asset.path))
    except Exception as error:
        raise DocumentIngestionError(f"Unable to render PDF: {asset.path}") from error


def _render_pages(document, native_pages, ocr_engine, dpi):
    pages: list[str] = []
    methods: list[TextExtractionMethod] = []
    for page_index, native_text in enumerate(native_pages):
        if native_text.strip():
            pages.append(native_text)
            methods.append(TextExtractionMethod.NATIVE)
            continue
        pages.append(_ocr_page(document, page_index, ocr_engine, dpi))
        methods.append(TextExtractionMethod.OCR)
    return pages, methods


def _ocr_page(document, page_index, ocr_engine, dpi):
    page = document.load_page(page_index)
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return ocr_engine.extract_text(image).strip()


def _extraction_method(methods):
    if all(method == TextExtractionMethod.OCR for method in methods):
        return TextExtractionMethod.OCR
    if all(method == TextExtractionMethod.NATIVE for method in methods):
        return TextExtractionMethod.NATIVE
    return TextExtractionMethod.HYBRID
