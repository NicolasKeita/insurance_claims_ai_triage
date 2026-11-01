"""Insurer application: teal masthead, shield and individually boxed fields."""

from evaluation.deterministic.document_generation.drawing import line, rectangle, text, title
from evaluation.deterministic.document_generation.stationery import BODY_FONT_SIZE, FOOTER, INK, MUTED

WHITE = (1, 1, 1)
MINT = (0.91, 0.97, 0.95)


def _header(page, style, heading):
    rectangle(page, (0, 0, 595, 143), style.accent)
    page.draw_polyline([(46, 37), (82, 37), (82, 65), (64, 82), (46, 65), (46, 37)], color=WHITE, width=1.8)
    page.draw_polyline([(54, 57), (62, 65), (75, 48)], color=WHITE, width=2.4)
    text(page, (101, 56), style.company, ("hebo", 18, WHITE), 440)
    text(page, (101, 78), style.tagline, ("helv", 9.5, WHITE), 440)
    text(page, (48, 116), style.contact, ("helv", 9, WHITE))
    title(page, (48, 198), heading)
    text(page, (48, 224), style.caption, ("hebo", 8.5, style.accent))


def _fields(page, rows, style, start):
    y = start
    for value in rows:
        section = value.endswith(":")
        item = ":" not in value
        if section:
            rectangle(page, (48, y - 19, 547, y + 11), style.accent)
            text(page, (60, y), value, ("hebo", BODY_FONT_SIZE, WHITE), 475)
        elif item:
            rectangle(page, (48, y - 19, 547, y + 11), MINT)
            text(page, (83, y), value, width=452)
        else:
            rectangle(page, (48, y - 21, 547, y + 12), MINT)
            line(page, (48, y + 12), (547, y + 12), style.accent)
            text(page, (60, y), value, ("helv", BODY_FONT_SIZE, INK), 475)
        y += 32 if section or item else 39


def _footer(page, style):
    text(page, (48, 718), style.note, ("helv", 9, MUTED))
    line(page, (48, 742), (547, 742), style.accent)
    for index, value in enumerate(FOOTER):
        text(page, (48, 763 + index * 18), value, ("helv", 8, MUTED))
    rectangle(page, (0, 825, 595, 842), style.accent)


def draw(page, lines, style, flags):
    header, footer = flags
    rows = lines[1:] if header else lines
    start = 270 if header else 66
    height = sum(32 if value.endswith(":") or ":" not in value else 39 for value in rows)
    if start + height > 691:
        raise ValueError("Business document exceeds the single-page printable area")
    if header:
        _header(page, style, lines[0])
    _fields(page, rows, style, start)
    if footer:
        _footer(page, style)
