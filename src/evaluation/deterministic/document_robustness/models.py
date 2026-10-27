"""Source-derived ground truth, constructed before any document transformation."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from claims.document_models import (
    AccidentReportExtraction,
    ClaimFormExtraction,
    DocumentSource,
    GarageQuoteExtraction,
    MoneyAmount,
    QuoteLineItem,
)
from claims.enums import CollisionType, DamageType, DocumentType

DocumentKind = Literal["claim_form", "accident_report", "garage_quote"]
StructuredDocument = ClaimFormExtraction | AccidentReportExtraction | GarageQuoteExtraction


@dataclass(frozen=True)
class ReferenceDocument:
    document_id: str
    document_type: DocumentKind
    text: str
    expected: StructuredDocument


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    documents: tuple[ReferenceDocument, ...]
    expected_consistency: dict


KINDS = ("claim_form", "accident_report", "garage_quote")
TYPES = (DocumentType.CLAIM_FORM, DocumentType.ACCIDENT_REPORT, DocumentType.REPAIR_QUOTE)


def build_expected(payload, quoted, report_date, garage):
    incident, vehicle = payload["incident"], payload["vehicle"]
    common = dict(
        claim_id=payload["claim_id"],
        location=incident["location"],
        vehicle_make=vehicle["make"],
        vehicle_model=vehicle["model"],
        collision_type=CollisionType.FRONT_COLLISION,
        injuries_declared=incident["injuries_declared"],
    )
    damage = tuple(DamageType(value) for value in payload["declared_damage"])
    sources = tuple(
        DocumentSource(document_type=kind, filename=f"{name}.pdf", page=1)
        for kind, name in zip(TYPES, KINDS, strict=True)
    )
    form = ClaimFormExtraction(
        **common,
        incident_date=date.fromisoformat(incident["date"]),
        vehicle_year=vehicle["year"],
        declared_damage=damage,
        source=sources[0],
    )
    report = AccidentReportExtraction(
        **common, incident_date=report_date, reported_damage=damage, passengers=1, source=sources[1]
    )
    quote = _expected_quote(payload, quoted, garage, sources[2])
    return form, report, quote


def _expected_quote(payload, quoted, garage, source):
    total = Decimal(payload["repair_estimate"]["amount"])
    currency = payload["repair_estimate"]["currency"]
    part_amount = (total / Decimal(len(quoted) + 1)).quantize(Decimal("0.01"))
    parts = tuple(
        QuoteLineItem(
            description=item.lower().replace("_", " "), price=MoneyAmount(amount=part_amount, currency=currency)
        )
        for item in quoted
    )
    return GarageQuoteExtraction(
        claim_id=payload["claim_id"],
        garage=garage,
        parts=parts,
        labor=MoneyAmount(amount=total - part_amount * len(parts), currency=currency),
        total=MoneyAmount(amount=total, currency=currency),
        source=source,
    )
