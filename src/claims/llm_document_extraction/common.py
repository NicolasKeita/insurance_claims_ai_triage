"""Shared conversions between LLM output and validated extraction models."""

from datetime import date
from decimal import Decimal

from claims.document_extraction import DocumentExtractionError
from claims.document_models import (
    AccidentReportExtraction,
    ClaimFormExtraction,
    GarageQuoteExtraction,
    MoneyAmount,
)

LlmDocumentExtraction = ClaimFormExtraction | AccidentReportExtraction | GarageQuoteExtraction


def to_money_amount(value) -> MoneyAmount:
    return MoneyAmount(amount=Decimal(str(value.amount)), currency=value.currency)


def parse_llm_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise DocumentExtractionError(f"Invalid LLM date: {value}") from error
