"""Distinct synthetic document layouts shared by demos and OCR fixtures."""

from evaluation.deterministic.document_generation import form, quote, report
from evaluation.deterministic.document_generation.stationery import (
    BODY_FONT_SIZE,
    STYLE_VERSION,
    STYLES,
    body_lines,
    body_rows,
    document_kind,
    visible_text,
)

__all__ = ["BODY_FONT_SIZE", "STYLE_VERSION", "render_page", "visible_text"]


def render_page(document, text, *, kind=None, header=True, footer=True):
    """Append one distinct A4 page, rejecting overflow and incomplete layouts."""
    name = document_kind(text, kind)
    lines = body_lines(text)
    title = lines[:1] if header else []
    body = body_rows(lines[1:] if header else lines)
    drawer = {"claim_form": form.draw, "garage_quote": quote.draw}.get(name, report.draw)
    page = document.new_page(width=595, height=842)
    try:
        drawer(page, title + body, STYLES[name], (header, footer))
    except ValueError:
        document.delete_page(len(document) - 1)
        raise
    metadata = dict(document.metadata or {})
    metadata.update(subject=f"{STYLE_VERSION}:{name}", creator="Insurance AI synthetic documents")
    document.set_metadata(metadata)
    return page
