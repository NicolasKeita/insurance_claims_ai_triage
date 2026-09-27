from dataclasses import dataclass

import pymupdf
from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from claims.assets import DocumentAsset
from claims.enums import TextExtractionMethod
from claims.ocr import OcrEngine


class DocumentIngestionError(Exception):
    pass


@dataclass(frozen=True)
class PdfTextExtraction:
    asset: DocumentAsset
    pages: tuple[str, ...]
    method: TextExtractionMethod = TextExtractionMethod.NATIVE

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def _has_pdf_signature(
    asset: DocumentAsset,
) -> bool:
    with asset.path.open("rb") as file:
        signature = file.read(5)

    return signature == b"%PDF-"


def extract_pdf_text(
    asset: DocumentAsset,
) -> PdfTextExtraction:
    if not _has_pdf_signature(asset):
        raise DocumentIngestionError(
            f"File is not a valid PDF: {asset.path}"
        )

    try:
        reader = PdfReader(
            str(asset.path),
            strict=False,
        )

    except PdfReadError as error:
        raise DocumentIngestionError(
            f"Unable to read PDF: {asset.path}"
        ) from error

    if reader.is_encrypted:
        raise DocumentIngestionError(
            f"Encrypted PDF is not supported: {asset.path}"
        )

    if len(reader.pages) == 0:
        raise DocumentIngestionError(
            f"PDF contains no pages: {asset.path}"
        )

    pages = tuple(
        (page.extract_text() or "").strip()
        for page in reader.pages
    )

    return PdfTextExtraction(
        asset=asset,
        pages=pages,
        method=TextExtractionMethod.NATIVE,
    )


def extract_pdf_text_auto(
    asset: DocumentAsset,
    ocr_engine: OcrEngine,
    dpi: int = 300,
) -> PdfTextExtraction:
    native = extract_pdf_text(asset)

    # Every page already contains usable native text:
    # OCR is unnecessary.
    if all(
        page.strip()
        for page in native.pages
    ):
        return native

    try:
        document = pymupdf.open(
            str(asset.path)
        )

    except Exception as error:
        raise DocumentIngestionError(
            f"Unable to render PDF: {asset.path}"
        ) from error

    pages: list[str] = []

    methods: list[
        TextExtractionMethod
    ] = []

    try:
        for page_index, native_text in enumerate(
            native.pages
        ):
            # Keep native text whenever available.
            if native_text.strip():
                pages.append(native_text)

                methods.append(
                    TextExtractionMethod.NATIVE
                )

                continue

            # No native text:
            # render this page and use OCR.
            page = document.load_page(
                page_index
            )

            pixmap = page.get_pixmap(
                dpi=dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (
                    pixmap.width,
                    pixmap.height,
                ),
                pixmap.samples,
            )

            ocr_text = (
                ocr_engine
                .extract_text(image)
                .strip()
            )

            pages.append(ocr_text)

            methods.append(
                TextExtractionMethod.OCR
            )

    finally:
        document.close()

    if all(
        method == TextExtractionMethod.OCR
        for method in methods
    ):
        extraction_method = (
            TextExtractionMethod.OCR
        )

    elif all(
        method == TextExtractionMethod.NATIVE
        for method in methods
    ):
        extraction_method = (
            TextExtractionMethod.NATIVE
        )

    else:
        extraction_method = (
            TextExtractionMethod.HYBRID
        )

    return PdfTextExtraction(
        asset=asset,
        pages=tuple(pages),
        method=extraction_method,
    )