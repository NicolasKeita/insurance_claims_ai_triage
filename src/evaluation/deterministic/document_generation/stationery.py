"""Reproducible branded synthetic documents, shared by demos and OCR fixtures.

Business lines remain visible, selectable and in their original reading order.
Stationery is deliberately extractable noise, separate from business sections.
"""

from dataclasses import dataclass
from textwrap import wrap

STYLE_VERSION = "professional_stationery_v2"
BODY_FONT_SIZE = 11.5
INK = (0.10, 0.17, 0.25)
MUTED = (0.33, 0.40, 0.46)
PALE = (0.95, 0.97, 0.98)
RULE = (0.82, 0.87, 0.90)


@dataclass(frozen=True)
class Stationery:
    company: str
    tagline: str
    contact: str
    caption: str
    note: str
    accent: tuple[float, float, float]


STYLES = {
    "claim_form": Stationery(
        "ALIZE ASSURANCES",
        "Motor insurance | Policyholder services",
        "18 Avenue des Amandiers, Bordeaux | contact@alize.example",
        "MOTOR CLAIMS / POLICYHOLDER DECLARATION",
        "Please retain a copy of this declaration for your records.",
        (0.03, 0.43, 0.47),
    ),
    "accident_report": Stationery(
        "ASTER EXPERTISE",
        "Independent automotive assessment office",
        "7 Rue des Ateliers, Bordeaux | accueil@aster.example",
        "INCIDENT RECORD / CIRCUMSTANCES AND OBSERVATIONS",
        "Prepared for document review. Signature: ____________________",
        (0.23, 0.32, 0.57),
    ),
    "garage_quote": Stationery(
        "ATELIER DES RIVES",
        "Automotive bodywork | Repair and maintenance",
        "42 Avenue du Port, Bordeaux | atelier@rives.example",
        "WORKSHOP ESTIMATE / SERVICE AND REPAIR DETAILS",
        "Work is subject to customer approval. No payment has been recorded.",
        (0.60, 0.30, 0.12),
    ),
    "police_report": Stationery(
        "BUREAU DES DOSSIERS",
        "Administrative document services",
        "9 Place des Archives, Bordeaux | dossiers@bureau.example",
        "SUPPORTING RECORD / ADMINISTRATIVE COPY",
        "Supporting record only. Keep with the corresponding claim documents.",
        (0.23, 0.37, 0.43),
    ),
}
FOOTER = (
    "Office hours: Monday to Friday, 09:00-18:00 | Correspondence by appointment",
    "Fictional organisation | Synthetic demonstration document | Reference copy",
)


def document_kind(text, kind):
    aliases = {"repair_quote": "garage_quote", "supporting_record": "police_report"}
    candidate = aliases.get(str(kind).lower(), str(kind).lower())
    if candidate in STYLES:
        return candidate
    title = text.splitlines()[0].upper() if text.strip() else ""
    for token, name in (
        ("CLAIM FORM", "claim_form"),
        ("ACCIDENT REPORT", "accident_report"),
        ("GARAGE QUOTE", "garage_quote"),
    ):
        if token in title:
            return name
    return "police_report"


def body_lines(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Cannot render an empty business document")
    return lines


def visible_text(text, *, kind=None, header=True, footer=True):
    """Source-derived reference including all stationery; no PDF parser involved."""
    style = STYLES[document_kind(text, kind)]
    body = body_lines(text)
    lines = [style.company, style.tagline, style.contact] if header else []
    if header:
        lines.extend((body[0], style.caption))
        lines.extend(body[1:])
    else:
        lines.extend(body)
    if footer:
        lines.extend((style.note, *FOOTER))
    return "\n".join(lines)


def body_rows(lines):
    """Wrap prose; keep structured business values on one readable line."""
    rows = []
    for line in lines:
        wrapped = wrap(line, width=82, break_long_words=False, break_on_hyphens=False)
        if len(wrapped) > 1 and ":" in line:
            raise ValueError(f"Business field exceeds the printable width: {line}")
        rows.extend(wrapped or [line])
    return rows
