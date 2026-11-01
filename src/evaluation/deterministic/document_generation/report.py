"""Independent expert's report: serif letterhead, compass seal and ruled notes."""

from evaluation.deterministic.document_generation.drawing import line, rectangle, text
from evaluation.deterministic.document_generation.stationery import BODY_FONT_SIZE, FOOTER, INK, MUTED


def _seal(page, accent):
    page.draw_circle((515, 63), 25, color=accent, width=1)
    page.draw_circle((515, 63), 19, color=accent, width=0.5)
    page.draw_polyline([(515, 43), (521, 63), (515, 83), (509, 63), (515, 43)], color=accent, width=1)
    line(page, (497, 63), (533, 63), accent)


def _header(page, style, heading):
    import pymupdf

    _seal(page, style.accent)
    text(page, (63, 55), style.company, ("tibo", 19, INK), 410)
    text(page, (63, 78), style.tagline, ("tiit", 10.5, MUTED), 410)
    text(page, (63, 112), style.contact, ("tiro", 9.5, MUTED), 480)
    line(page, (63, 128), (543, 128), INK)
    size = min(27, 480 / max(1, pymupdf.get_text_length(heading, fontname="tibo", fontsize=1)))
    if size < 12:
        raise ValueError("Document title exceeds the printable width")
    text(page, (63, 177), heading, ("tibo", size, INK), 480)
    text(page, (63, 201), style.caption, ("cour", 8, style.accent), 480)


def _observations(page, rows, style, start):
    y = start
    line(page, (81, start - 16), (81, 650), style.accent)
    for value in rows:
        section = value.endswith(":")
        y += 11 if section else 0
        if section:
            line(page, (104, y - 17), (543, y - 17))
        font = "tibo" if section else "tiro"
        if value.startswith("Passengers:"):
            font = "helv"
        text(page, (104, y), value, (font, BODY_FONT_SIZE, INK), 439)
        y += 27


def _footer(page, style):
    text(page, (104, 700), style.note, ("tiit", 9.5, MUTED), 439)
    line(page, (63, 739), (543, 739), INK)
    for index, value in enumerate(FOOTER):
        text(page, (63, 760 + index * 18), value, ("tiro", 8.5, MUTED), 480)
    rectangle(page, (63, 810, 543, 814), style.accent)


def draw(page, lines, style, flags):
    header, footer = flags
    rows = lines[1:] if header else lines
    start = 250 if header else 76
    height = len(rows) * 27 + sum(11 for value in rows if value.endswith(":"))
    if start + height > 670:
        raise ValueError("Business document exceeds the single-page printable area")
    if header:
        _header(page, style, lines[0])
    _observations(page, rows, style, start)
    if footer:
        _footer(page, style)
