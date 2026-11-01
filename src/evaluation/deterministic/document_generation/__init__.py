"""Reproducible branded synthetic PDFs shared by demos and OCR fixtures."""

from evaluation.deterministic.document_generation.stationery import (
    BODY_FONT_SIZE,
    FOOTER,
    INK,
    MUTED,
    PALE,
    RULE,
    STYLE_VERSION,
    STYLES,
    body_lines,
    body_rows,
    document_kind,
    visible_text,
)

__all__ = ["BODY_FONT_SIZE", "STYLE_VERSION", "render_page", "visible_text"]


def _text(page, point, value, size=BODY_FONT_SIZE, style=(False, INK)):
    import pymupdf

    bold, color = style
    font = "hebo" if bold else "helv"
    if pymupdf.get_text_length(value, fontname=font, fontsize=size) > 490:
        raise ValueError(f"Document line exceeds the printable width: {value}")
    page.insert_text(point, value, fontname=font, fontsize=size, color=color)


def _logo(page, accent):
    import pymupdf

    page.draw_rect(pymupdf.Rect(46, 40, 82, 79), color=accent, fill=accent)
    white = (1, 1, 1)
    page.draw_polyline([(53, 69), (64, 49), (75, 69)], color=white, width=2.4)
    page.draw_line((58, 62), (70, 62), color=white, width=2.2)
    page.draw_line((47, 85), (81, 85), color=accent, width=3)


def _header(page, style, title):
    import pymupdf

    _logo(page, style.accent)
    _text(page, (98, 57), style.company, 16, style=(True, INK))
    _text(page, (98, 77), style.tagline, 9, style=(False, MUTED))
    _text(page, (46, 108), style.contact, 9, style=(False, MUTED))
    page.draw_line((46, 122), (549, 122), color=RULE, width=0.7)
    title_size = min(21, 490 / max(1, pymupdf.get_text_length(title, fontname="hebo", fontsize=1)))
    if title_size < 12:
        raise ValueError("Document title exceeds the printable width")
    _text(page, (46, 164), title, title_size, style=(True, INK))
    _text(page, (46, 187), style.caption, 8.5, style=(True, style.accent))


def _body(page, lines, style, *, header, quote):
    import pymupdf

    rows = body_rows(lines)
    start = 230 if header else 72
    step = 25 if quote else 24
    if start + max(0, len(rows) - 1) * step > 666:
        raise ValueError("Business document exceeds the single-page printable area")
    bottom = start + max(0, len(rows) - 1) * step + 17
    page.draw_rect(pymupdf.Rect(46, start - 20, 549, bottom), color=RULE, width=0.6)
    for index, line in enumerate(rows):
        y = start + index * step
        total = line.lower().startswith("total:")
        section = line.endswith(":")
        if total or section or (quote and index % 2 == 0):
            page.draw_rect(pymupdf.Rect(47, y - 16, 548, y + 8), color=None, fill=PALE)
        _text(page, (59, y), line, style=(total or section, style.accent if total else INK))


def _footer(page, style):
    _text(page, (46, 716), style.note, 9, style=(False, MUTED))
    page.draw_line((46, 742), (549, 742), color=RULE, width=0.7)
    for index, line in enumerate(FOOTER):
        _text(page, (46, 762 + index * 18), line, 8.5, style=(False, MUTED))


def render_page(document, text, *, kind=None, header=True, footer=True):
    """Append one A4 page; reject content exceeding the printable area."""
    name = document_kind(text, kind)
    style = STYLES[name]
    lines = body_lines(text)
    body = lines[1:] if header else lines
    rows = body_rows(body)
    start, step = (230 if header else 72), (25 if name == "garage_quote" else 24)
    if start + max(0, len(rows) - 1) * step > 666:
        raise ValueError("Business document exceeds the single-page printable area")
    page = document.new_page(width=595, height=842)
    if header:
        _header(page, style, lines[0])
    _body(page, body, style, header=header, quote=name == "garage_quote")
    if footer:
        _footer(page, style)
    return page
