from dataclasses import dataclass

from claims.assets import DocumentAsset

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentIngestionError(Exception):
    pass

@dataclass(frozen=True)
class PdfTextExtraction:
    asset: DocumentAsset
    pages: tuple[str, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

def _has_pdf_signature(asset: DocumentAsset) -> bool:
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

    pages = tuple(
        (page.extract_text() or "").strip()
        for page in reader.pages
    )

    return PdfTextExtraction(
        asset=asset,
        pages=pages,
    )