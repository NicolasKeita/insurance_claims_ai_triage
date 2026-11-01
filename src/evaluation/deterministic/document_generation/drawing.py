"""Small drawing primitives; PDF libraries remain lazy dependencies."""

from evaluation.deterministic.document_generation.stationery import BODY_FONT_SIZE, INK, RULE


def text(page, point, value, style=("helv", BODY_FONT_SIZE, INK), width=490):
    import pymupdf

    font, size, color = style
    if pymupdf.get_text_length(value, fontname=font, fontsize=size) > width + 0.01:
        raise ValueError(f"Document line exceeds the printable width: {value}")
    page.insert_text(point, value, fontname=font, fontsize=size, color=color)


def title(page, point, value, width=490):
    import pymupdf

    size = min(25, width / max(1, pymupdf.get_text_length(value, fontname="hebo", fontsize=1)))
    if size < 12:
        raise ValueError("Document title exceeds the printable width")
    text(page, point, value, ("hebo", size, INK), width)


def rectangle(page, box, fill=None, color=None):
    import pymupdf

    page.draw_rect(pymupdf.Rect(box), color=color, fill=fill, width=0.6)


def line(page, start, end, color=RULE):
    page.draw_line(start, end, color=color, width=0.7)
