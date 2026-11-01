"""Workshop estimate stationery with a charcoal masthead and aligned repair costs."""

import math

from evaluation.deterministic.document_generation.drawing import line, rectangle, text, title
from evaluation.deterministic.document_generation.stationery import FOOTER, INK, MUTED, RULE

CHARCOAL = (0.12, 0.15, 0.17)
ORANGE = (0.84, 0.34, 0.10)
CREAM = (0.98, 0.96, 0.93)
WHITE = (1, 1, 1)


def _wheel(page):
    page.draw_circle((520, 68), 29, color=ORANGE, width=3)
    page.draw_circle((520, 68), 12, color=WHITE, width=2)
    for angle in range(0, 360, 45):
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        line(page, (520 + 13 * dx, 68 + 13 * dy), (520 + 24 * dx, 68 + 24 * dy), ORANGE)


def _header(page, lines, style):
    rectangle(page, (0, 0, 595, 151), fill=CHARCOAL)
    _wheel(page)
    text(page, (48, 54), style.company, ("hebo", 18, WHITE), width=420)
    text(page, (48, 78), style.tagline, ("helv", 9, WHITE), width=420)
    text(page, (48, 115), style.contact, ("helv", 9, WHITE))
    title(page, (48, 202), lines[0], width=500)
    text(page, (48, 228), style.caption, ("hebo", 8.5, ORANGE))


def _plan(body, header):
    lead = 0
    for value in body:
        if not value.startswith(("Garage:", "Claim ID:")):
            break
        lead += 1
    start = 274 if header else 70
    first_row = start + lead * 28 + (35 if lead else 0)
    end = first_row + sum(51 if value.lower().startswith("total:") else 35 for value in body[lead:])
    if end > 672:
        raise ValueError("Business document exceeds the single-page printable area")
    return lead, start, first_row


def _amount_row(page, value, y, alternate):
    total = value.lower().startswith("total:")
    if total:
        rectangle(page, (277, y - 16, 553, y + 22), fill=CREAM, color=ORANGE)
        text(page, (291, y + 9), value, ("hebo", 16, INK), width=248)
        return 51
    if alternate:
        rectangle(page, (48, y - 15, 553, y + 13), fill=CREAM)
    label, separator, amount = value.partition(":")
    if separator and amount.strip():
        import pymupdf

        text(page, (58, y + 4), f"{label}: ", ("cour", 11.5, INK), width=340)
        x = 530 - pymupdf.get_text_length(amount.strip(), fontname="cour", fontsize=11.5)
        text(page, (x, y + 4), amount.strip(), ("cour", 11.5, INK), width=100)
    else:
        text(page, (58, y + 4), value, ("cour", 11.5, INK), width=485)
    line(page, (48, y + 14), (553, y + 14), RULE)
    return 35


def _body(page, body, plan):
    lead, start, y = plan
    if lead:
        rectangle(page, (48, start - 21, 553, start + lead * 28 - 8), fill=CREAM)
    for index, value in enumerate(body[:lead]):
        text(page, (61, start + index * 28), value, ("hebo", 10.5, INK))
    for index, value in enumerate(body[lead:]):
        y += _amount_row(page, value, y, index % 2 == 0)


def _footer(page, style):
    text(page, (48, 716), style.note, ("helv", 8.5, MUTED))
    line(page, (48, 739), (553, 739), ORANGE)
    for index, value in enumerate(FOOTER):
        text(page, (48, 759 + index * 17), value, ("helv", 7.5, MUTED))


def draw(page, lines, style, flags):
    """Render the complete original body without changing financial information."""
    header, footer = flags
    body = lines[1:] if header else lines
    plan = _plan(body, header)
    rectangle(page, (0, 0, 15, 842), fill=ORANGE)
    if header:
        _header(page, lines, style)
    _body(page, body, plan)
    if footer:
        _footer(page, style)
