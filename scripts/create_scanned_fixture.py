from pathlib import Path

import pymupdf


SOURCE = Path(
    "data/CLAIM-2026-00001/documents/accident_report.pdf"
)

TARGET = Path(
    "tests/data/ocr/accident_report_scan.pdf"
)


TARGET.parent.mkdir(
    parents=True,
    exist_ok=True,
)


source_document = pymupdf.open(
    SOURCE
)

output_document = pymupdf.open()


for source_page in source_document:
    pixmap = source_page.get_pixmap(
        dpi=200,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    output_page = output_document.new_page(
        width=source_page.rect.width,
        height=source_page.rect.height,
    )

    output_page.insert_image(
        output_page.rect,
        pixmap=pixmap,
    )


output_document.save(
    TARGET,
    garbage=4,
    deflate=True,
)

output_document.close()
source_document.close()


print(f"Created image-only PDF: {TARGET}")