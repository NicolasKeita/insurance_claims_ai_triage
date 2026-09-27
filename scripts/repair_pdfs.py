from pathlib import Path
from tempfile import NamedTemporaryFile

from pypdf import PdfReader, PdfWriter


DOCUMENTS_DIR = Path(
    "data/CLAIM-2026-00001/documents"
)


def repair_pdf(path: Path) -> None:
    reader = PdfReader(
        str(path),
        strict=False,
    )

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    with NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
        dir=path.parent,
    ) as tmp:
        temp_path = Path(tmp.name)

        writer.write(tmp)

    temp_path.replace(path)

    print(f"Rewritten: {path}")


for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
    repair_pdf(pdf_path)