"""Deterministic repair-quote extraction."""

from decimal import Decimal

from claims.document_extraction.base import (
    DAMAGE_LABEL_MAP,
    MONEY_LINE_PATTERN,
    DocumentExtractionError,
    find_prefixed_value,
    text_lines,
)
from claims.document_models import DocumentSource, GarageQuoteExtraction, MoneyAmount, QuoteLineItem
from claims.enums import DocumentType


def extract_garage_quote(extraction) -> GarageQuoteExtraction:
    if extraction.asset.type != DocumentType.REPAIR_QUOTE:
        raise DocumentExtractionError("Expected a repair quote document")
    if not extraction.has_text:
        raise DocumentExtractionError("Repair quote contains no extractable text")
    lines = text_lines(extraction)
    parts, labor, total = _quote_amounts(lines)
    _validate_quote_totals(parts, labor, total)
    return GarageQuoteExtraction(
        garage=find_prefixed_value(lines, "Garage:"),
        claim_id=find_prefixed_value(lines, "Claim ID:"),
        parts=tuple(parts),
        labor=labor,
        total=total,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )


def _quote_amounts(lines):
    parts: list[QuoteLineItem] = []
    labor: MoneyAmount | None = None
    total: MoneyAmount | None = None
    for line in lines:
        match = MONEY_LINE_PATTERN.match(line)
        if match is None:
            label = line.partition(":")[0].strip().lower()
            if label in {*DAMAGE_LABEL_MAP, "labor", "total"}:
                raise DocumentExtractionError(f"Invalid quote amount: {line}")
            continue
        label = match.group("label").strip()
        money = MoneyAmount(amount=Decimal(match.group("amount").replace(",", ".")), currency=match.group("currency"))
        if label.lower() == "labor":
            if labor is not None and labor != money:
                raise DocumentExtractionError("Conflicting labor amounts")
            labor = money
        elif label.lower() == "total":
            if total is not None and total != money:
                raise DocumentExtractionError("Conflicting total amounts")
            total = money
        else:
            parts.append(QuoteLineItem(description=label, price=money))
    if labor is None:
        raise DocumentExtractionError("Missing labor amount")
    if total is None:
        raise DocumentExtractionError("Missing total amount")
    return parts, labor, total


def _validate_quote_totals(parts, labor, total):
    currencies = {item.price.currency for item in parts}
    currencies.add(labor.currency)
    currencies.add(total.currency)
    if len(currencies) != 1:
        raise DocumentExtractionError("Quote contains inconsistent currencies")
    calculated_total = sum((item.price.amount for item in parts), Decimal("0")) + labor.amount
    if calculated_total != total.amount:
        raise DocumentExtractionError("Quote total does not match line items")
