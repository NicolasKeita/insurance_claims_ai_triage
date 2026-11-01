"""Distinct document identities remain legible without changing their evidence."""

import pymupdf
import pytest

from evaluation.deterministic.document_generation import render_page
from tests.infrastructure.evaluation.documents._pdf_documents import FORM, QUOTE, REPORT

DOCUMENTS = (("claim_form", FORM), ("accident_report", REPORT), ("garage_quote", QUOTE))


def _header_geometry(page):
    """Ignore colours and text so recolouring one template cannot pass."""
    return tuple(
        (
            path["type"],
            tuple(round(value, 2) for value in path["rect"]),
            tuple(item[0] for item in path["items"]),
        )
        for path in page.get_drawings()
        if path["rect"].y0 < 180
    )


def test_three_document_types_have_distinct_geometric_page_headers():
    geometries = []
    for kind, text in DOCUMENTS:
        with pymupdf.open() as document:
            page = render_page(document, text, kind=kind)
            geometries.append(_header_geometry(page))
    assert all(geometries)
    assert len(set(geometries)) == len(DOCUMENTS)


@pytest.mark.parametrize("kind,text", DOCUMENTS, ids=("form", "report", "quote"))
def test_all_visible_text_stays_inside_the_page_and_does_not_overlap(kind, text):
    with pymupdf.open() as document:
        page = render_page(document, text, kind=kind)
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip()
        ]
        assert spans
        for span in spans:
            bounds = pymupdf.Rect(span["bbox"])
            assert page.rect.contains(bounds), span["text"]
        for index, span in enumerate(spans):
            for other in spans[index + 1 :]:
                intersection = pymupdf.Rect(span["bbox"]) & pymupdf.Rect(other["bbox"])
                assert intersection.get_area() < 2, (span["text"], other["text"])
