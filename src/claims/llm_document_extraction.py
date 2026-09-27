from pydantic import ValidationError
from decimal import Decimal
from claims.document_models import (
    DocumentSource,
    GarageQuoteExtraction,
    GarageQuoteLlmOutput,
    MoneyAmount,
    QuoteLineItem,
)

from claims.document_extraction import (
    DocumentExtractionError,
)
from claims.document_ingestion import (
    PdfTextExtraction,
)
from claims.document_models import (
    DocumentSource,
    GarageQuoteExtraction,
    GarageQuoteLlmOutput,
)
from claims.enums import DocumentType
from claims.llm import StructuredLlm

def build_garage_quote_prompt(
    extraction: PdfTextExtraction,
) -> str:
    return f"""
Extract ALL explicitly stated information from this insurance repair quote.

Return the result as structured JSON matching the provided schema.

Important mapping rules:
- "Garage:" maps to garage.
- "Claim ID:" maps to claim_id.
- Repair/replacement lines map to parts.
- "Labor:" maps to labor.
- "Total:" maps to total.
- Labor must NOT be included in parts.
- Total must NOT be included in parts.
- Monetary "amount" must contain ONLY the numeric value.
- Correct example: {{"amount": 780, "currency": "EUR"}}
- Never include the currency inside the amount field.
- If a field is explicitly present in the document, you MUST extract it.
- Return null ONLY when the information is genuinely absent.
- Do not invent missing information.

DOCUMENT:

{extraction.text}
""".strip()

def extract_garage_quote_with_llm(
    extraction: PdfTextExtraction,
    llm: StructuredLlm,
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
    result = llm.generate(
        prompt=build_garage_quote_prompt(
            extraction
        ),
        response_model=GarageQuoteLlmOutput,
    )

    missing_fields = []

    if result.garage is None:
        missing_fields.append("garage")

    if result.claim_id is None:
        missing_fields.append("claim_id")

    if result.labor is None:
        missing_fields.append("labor")

    if result.total is None:
        missing_fields.append("total")

    if missing_fields:
        raise DocumentExtractionError(
            "Missing required quote fields: "
            + ", ".join(missing_fields)
        )

    parts = tuple(
        QuoteLineItem(
            description=item.description,
            price=_to_money_amount(
                item.price
            ),
        )
        for item in result.parts
    )
    labor = _to_money_amount(
        result.labor
    )

    total = _to_money_amount(
        result.total
    )

    try:
        return GarageQuoteExtraction(
            garage=result.garage,
            claim_id=result.claim_id,
            parts=parts,
            labor=labor,
            total=total,
            source=DocumentSource(
                document_type=extraction.asset.type,
                filename=extraction.asset.filename,
                page=1,
            ),
        )

    except ValidationError as error:
        raise DocumentExtractionError(
            "LLM produced an invalid garage quote"
        ) from error

def _to_money_amount(
    value,
) -> MoneyAmount:
    return MoneyAmount(
        amount=Decimal(str(value.amount)),
        currency=value.currency,
    )