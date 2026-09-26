import re
from decimal import Decimal

from claims.document_ingestion import PdfTextExtraction
from claims.document_models import (
    DocumentSource,
    GarageQuoteExtraction,
    MoneyAmount,
    QuoteLineItem,
)
from claims.enums import DocumentType

class DocumentExtractionError(Exception):
    pass

MONEY_LINE_PATTERN = re.compile(
    r"^(?P<label>[^:]+):\s*"
    r"(?P<amount>\d+(?:[.,]\d{1,2})?)\s+"
    r"(?P<currency>[A-Z]{3})$"
)

def _find_prefixed_value(
    lines: list[str],
    prefix: str,
) -> str:
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix):].strip()

            if value:
                return value

    raise DocumentExtractionError(
        f"Missing field: {prefix}"
    )

def extract_garage_quote(
    extraction: PdfTextExtraction,
) -> GarageQuoteExtraction:
    if (
        extraction.asset.type
        != DocumentType.REPAIR_QUOTE
    ):
        raise DocumentExtractionError(
            "Expected a repair quote document"
        )
    
    if not extraction.has_text:
        raise DocumentExtractionError(
            "Repair quote contains no extractable text"
        )
    
    lines = [
        line.strip()
        for line in extraction.text.splitlines()
        if line.strip()
    ]

    garage = _find_prefixed_value(
        lines,
        "Garage:",
    )

    claim_id = _find_prefixed_value(
        lines,
        "Claim ID:",
    )
    parts: list[QuoteLineItem] = []

    labor: MoneyAmount | None = None
    total: MoneyAmount | None = None

    for line in lines:
        match = MONEY_LINE_PATTERN.match(line)

        if match is None:
            continue

        label = match.group("label").strip()

        amount = Decimal(
            match.group("amount").replace(",", ".")
        )

        currency = match.group("currency")

        money = MoneyAmount(
            amount=amount,
            currency=currency,
        )
        if label.lower() == "labor":
            labor = money

        elif label.lower() == "total":
            total = money

        else:
            parts.append(
                QuoteLineItem(
                    description=label,
                    price=money,
                )
            )

    if labor is None:
        raise DocumentExtractionError(
            "Missing labor amount"
        )

    if total is None:
        raise DocumentExtractionError(
            "Missing total amount"
        )
    currencies = {
        item.price.currency
        for item in parts
    }

    currencies.add(labor.currency)
    currencies.add(total.currency)

    if len(currencies) != 1:
        raise DocumentExtractionError(
            "Quote contains inconsistent currencies"
        )
    calculated_total = sum(
        (
            item.price.amount
            for item in parts
        ),
        Decimal("0"),
    )

    calculated_total += labor.amount
    if calculated_total != total.amount:
        raise DocumentExtractionError(
            "Quote total does not match line items"
        )
    return GarageQuoteExtraction(
        garage=garage,
        claim_id=claim_id,
        parts=tuple(parts),
        labor=labor,
        total=total,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )