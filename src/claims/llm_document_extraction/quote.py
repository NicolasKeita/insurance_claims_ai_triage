"""LLM-assisted repair-quote extraction with a single targeted retry."""

from decimal import Decimal

from pydantic import ValidationError

from claims.document_extraction import DocumentExtractionError
from claims.document_models import DocumentSource, GarageQuoteExtraction, GarageQuoteLlmOutput, QuoteLineItem
from claims.enums import DocumentType
from claims.llm import StructuredLlm
from claims.llm_document_extraction.common import to_money_amount
from claims.llm_document_extraction.prompts import build_garage_quote_prompt, build_garage_quote_retry_prompt


def missing_quote_fields(result: GarageQuoteLlmOutput) -> list[str]:
    missing_fields = []
    if result.garage is None:
        missing_fields.append("garage")
    if result.claim_id is None:
        missing_fields.append("claim_id")
    if result.labor is None:
        missing_fields.append("labor")
    if result.total is None:
        missing_fields.append("total")
    return missing_fields


def extract_garage_quote_with_llm(extraction, llm: StructuredLlm) -> GarageQuoteExtraction:
    if extraction.asset.type != DocumentType.REPAIR_QUOTE:
        raise DocumentExtractionError("Expected a repair quote document")
    if not extraction.has_text:
        raise DocumentExtractionError("Repair quote contains no extractable text")
    result = llm.generate(prompt=build_garage_quote_prompt(extraction), response_model=GarageQuoteLlmOutput)
    missing_fields = missing_quote_fields(result)
    if missing_fields:
        result = _retry_quote(extraction, llm, result, missing_fields)
    return _build_quote(extraction, result)


def _retry_quote(extraction, llm, initial_result, missing_fields):
    result = llm.generate(
        prompt=build_garage_quote_retry_prompt(extraction, missing_fields),
        response_model=GarageQuoteLlmOutput,
    )
    remaining = missing_quote_fields(result)
    if remaining:
        raise DocumentExtractionError(
            "Missing required quote fields after retry: "
            + ", ".join(remaining)
            + "\n\nINITIAL STRUCTURED LLM OUTPUT:\n"
            + initial_result.model_dump_json(indent=2)
            + "\n\nRETRY STRUCTURED LLM OUTPUT:\n"
            + result.model_dump_json(indent=2)
        )
    return result


def _build_quote(extraction, result):
    parts = tuple(
        QuoteLineItem(description=item.description, price=to_money_amount(item.price)) for item in result.parts
    )
    try:
        return GarageQuoteExtraction(
            garage=result.garage,
            claim_id=result.claim_id,
            parts=parts,
            labor=to_money_amount(result.labor),
            total=to_money_amount(result.total),
            source=DocumentSource(
                document_type=extraction.asset.type,
                filename=extraction.asset.filename,
                page=1,
            ),
        )
    except ValidationError as error:
        raise DocumentExtractionError(_invalid_quote_message(result, error)) from error


def _invalid_quote_message(result, error):
    parts_total = sum((item.price.amount for item in result.parts), Decimal("0"))
    labor_amount = Decimal(str(result.labor.amount)) if result.labor is not None else None
    declared_total = Decimal(str(result.total.amount)) if result.total is not None else None
    calculated_total = parts_total + labor_amount if labor_amount is not None else None
    return (
        "LLM produced an invalid garage quote.\n\n"
        f"STRUCTURED LLM OUTPUT:\n"
        f"{result.model_dump_json(indent=2)}\n\n"
        f"PARTS TOTAL: {parts_total}\n"
        f"LABOR: {labor_amount}\n"
        f"CALCULATED TOTAL: {calculated_total}\n"
        f"DECLARED TOTAL: {declared_total}\n\n"
        f"VALIDATION ERROR:\n{error}"
    )
