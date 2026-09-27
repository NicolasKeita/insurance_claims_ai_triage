from datetime import date
from decimal import Decimal

from pydantic import ValidationError

from claims.document_extraction import (
    DocumentExtractionError,
)
from claims.document_ingestion import (
    PdfTextExtraction,
)
from claims.document_models import (
    AccidentReportExtraction,
    AccidentReportLlmOutput,
    ClaimFormExtraction,
    ClaimFormLlmOutput,
    DocumentSource,
    GarageQuoteExtraction,
    GarageQuoteLlmOutput,
    MoneyAmount,
    QuoteLineItem,
)
from claims.enums import DocumentType
from claims.llm import StructuredLlm

def build_garage_quote_prompt(
    extraction: PdfTextExtraction,
) -> str:
    return f"""
Extract ALL explicitly stated information from this repair quote.

Return structured JSON matching the provided schema.

Rules:
- "Garage:" maps to garage.
- "Claim ID:" maps to claim_id.
- EVERY repair or replacement line before "Labor:" must be included in parts.
- For each part:
  - description = the text before ":"
  - price.amount = numeric amount only
  - price.currency = three-letter currency code
- "Labor:" maps ONLY to labor.
- "Total:" maps ONLY to total.
- Labor must NOT be included in parts.
- Total must NOT be included in parts.
- DO NOT omit repair lines.
- If the document contains:
    Front bumper: 780 EUR
    Left headlight: 620 EUR
    Hood: 840 EUR
  then parts MUST contain exactly 3 entries.
- Monetary amount must contain only the numeric value.
- Correct: {{"amount": 780, "currency": "EUR"}}
- Incorrect: {{"amount": "780 EUR", "currency": "EUR"}}
- Return null only when a scalar field is genuinely absent.
- Never invent missing information.

DOCUMENT:

{extraction.text}
""".strip()

def build_garage_quote_retry_prompt(
    extraction: PdfTextExtraction,
    missing_fields: list[str],
) -> str:
    missing = ", ".join(missing_fields)

    return f"""
The previous extraction missed these fields:

{missing}

Re-read the repair quote carefully and extract the complete structured result.

Important:
- Look at EVERY line of the document.
- "Garage:" maps to garage.
- "Claim ID:" maps to claim_id.
- "Labor:" maps to labor.
- "Total:" maps to total.
- Repair/replacement lines map to parts.
- Labor must NOT be included in parts.
- Total must NOT be included in parts.
- Monetary amount must contain ONLY the numeric value.
- Example: {{"amount": 920, "currency": "EUR"}}
- Never include "EUR" inside amount.
- If a requested field is explicitly present, you MUST extract it.
- Return null ONLY if the information is genuinely absent.
- Do not calculate or invent missing values.
- Return valid JSON matching the provided schema.

DOCUMENT:

{extraction.text}
""".strip()

def _missing_quote_fields(
    result: GarageQuoteLlmOutput,
) -> list[str]:
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

    missing_fields = _missing_quote_fields(result)

    if missing_fields:
        initial_result = result
        result = llm.generate(
            prompt=build_garage_quote_retry_prompt(
                extraction,
                missing_fields,
            ),
            response_model=GarageQuoteLlmOutput,
        )
        missing_fields = _missing_quote_fields(result)
        if missing_fields:
            raise DocumentExtractionError(
                "Missing required quote fields after retry: "
                + ", ".join(missing_fields)
                + "\n\nINITIAL STRUCTURED LLM OUTPUT:\n"
                + initial_result.model_dump_json(indent=2)
                + "\n\nRETRY STRUCTURED LLM OUTPUT:\n"
                + result.model_dump_json(indent=2)
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
        parts_total = sum(
            (
                item.price.amount
                for item in result.parts
            ),
            Decimal("0"),
        )

        labor_amount = (
            Decimal(str(result.labor.amount))
            if result.labor is not None
            else None
        )

        declared_total = (
            Decimal(str(result.total.amount))
            if result.total is not None
            else None
        )

        calculated_total = (
            parts_total + labor_amount
            if labor_amount is not None
            else None
        )

        raise DocumentExtractionError(
            "LLM produced an invalid garage quote.\n\n"
            f"STRUCTURED LLM OUTPUT:\n"
            f"{result.model_dump_json(indent=2)}\n\n"
            f"PARTS TOTAL: {parts_total}\n"
            f"LABOR: {labor_amount}\n"
            f"CALCULATED TOTAL: {calculated_total}\n"
            f"DECLARED TOTAL: {declared_total}\n\n"
            f"VALIDATION ERROR:\n{error}"
        ) from error

def _to_money_amount(
    value,
) -> MoneyAmount:
    return MoneyAmount(
        amount=Decimal(str(value.amount)),
        currency=value.currency,
    )

def _parse_llm_date(
    value: str,
) -> date:
    try:
        return date.fromisoformat(value)

    except ValueError as error:
        raise DocumentExtractionError(
            f"Invalid LLM date: {value}"
        ) from error


def build_claim_form_prompt(
    extraction: PdfTextExtraction,
) -> str:
    return f"""
Extract ALL explicitly stated information from this insurance claim form.

Rules:
- Use only information present in the document.
- Never invent missing information.
- Return null when a requested scalar field is absent.
- Use YYYY-MM-DD for incident_date.
- Return vehicle make and model separately.
- declared_damage must contain only damage types supported by the schema.
- injuries_declared must be true or false only when explicitly stated.
- Preserve the claim ID exactly.

DOCUMENT:

{extraction.text}
""".strip()

def extract_claim_form_with_llm(
    extraction: PdfTextExtraction,
    llm: StructuredLlm,
) -> ClaimFormExtraction:
    if (
        extraction.asset.type
        != DocumentType.CLAIM_FORM
    ):
        raise DocumentExtractionError(
            "Expected a claim form document"
        )

    if not extraction.has_text:
        raise DocumentExtractionError(
            "Claim form contains no extractable text"
        )
    result = llm.generate(
        prompt=build_claim_form_prompt(
            extraction
        ),
        response_model=ClaimFormLlmOutput,
    )
    required_fields = {
        "claim_id": result.claim_id,
        "incident_date": result.incident_date,
        "location": result.location,
        "vehicle_make": result.vehicle_make,
        "vehicle_model": result.vehicle_model,
        "vehicle_year": result.vehicle_year,
        "collision_type": result.collision_type,
        "injuries_declared": result.injuries_declared,
    }

    missing_fields = [
        field
        for field, value in required_fields.items()
        if value is None
    ]

    if missing_fields:
        raise DocumentExtractionError(
            "Missing required claim form fields: "
            + ", ".join(missing_fields)
        )
    return ClaimFormExtraction(
        claim_id=result.claim_id,
        incident_date=_parse_llm_date(
            result.incident_date
        ),
        location=result.location,
        vehicle_make=result.vehicle_make,
        vehicle_model=result.vehicle_model,
        vehicle_year=result.vehicle_year,
        collision_type=result.collision_type,
        declared_damage=result.declared_damage,
        injuries_declared=result.injuries_declared,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )

def build_accident_report_prompt(
    extraction: PdfTextExtraction,
) -> str:
    return f"""
Extract ALL explicitly stated information from this accident report.

Rules:
- Use only information present in the document.
- Never invent missing information.
- Return null when a requested scalar field is absent.
- Use YYYY-MM-DD for incident_date.
- Return vehicle make and model separately.
- reported_damage must contain only damage types supported by the schema.
- passengers must be the explicitly stated number of passengers.
- injuries_declared must be true or false only when explicitly stated.
- Preserve the claim ID exactly.

DOCUMENT:

{extraction.text}
""".strip()

def extract_accident_report_with_llm(
    extraction: PdfTextExtraction,
    llm: StructuredLlm,
) -> AccidentReportExtraction:
    if (
        extraction.asset.type
        != DocumentType.ACCIDENT_REPORT
    ):
        raise DocumentExtractionError(
            "Expected an accident report document"
        )

    if not extraction.has_text:
        raise DocumentExtractionError(
            "Accident report contains no extractable text"
        )
    result = llm.generate(
        prompt=build_accident_report_prompt(
            extraction
        ),
        response_model=AccidentReportLlmOutput,
    )
    required_fields = {
        "claim_id": result.claim_id,
        "incident_date": result.incident_date,
        "location": result.location,
        "vehicle_make": result.vehicle_make,
        "vehicle_model": result.vehicle_model,
        "collision_type": result.collision_type,
        "passengers": result.passengers,
        "injuries_declared": result.injuries_declared,
    }

    missing_fields = [
        field
        for field, value in required_fields.items()
        if value is None
    ]

    if missing_fields:
        raise DocumentExtractionError(
            "Missing required accident report fields: "
            + ", ".join(missing_fields)
        )
    return AccidentReportExtraction(
        claim_id=result.claim_id,
        incident_date=_parse_llm_date(
            result.incident_date
        ),
        location=result.location,
        vehicle_make=result.vehicle_make,
        vehicle_model=result.vehicle_model,
        collision_type=result.collision_type,
        reported_damage=result.reported_damage,
        passengers=result.passengers,
        injuries_declared=result.injuries_declared,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )

LlmDocumentExtraction = (
    ClaimFormExtraction
    | AccidentReportExtraction
    | GarageQuoteExtraction
)

def extract_document_with_llm(
    extraction: PdfTextExtraction,
    llm: StructuredLlm,
) -> LlmDocumentExtraction:
    match extraction.asset.type:
        case DocumentType.CLAIM_FORM:
            return extract_claim_form_with_llm(
                extraction,
                llm,
            )

        case DocumentType.ACCIDENT_REPORT:
            return extract_accident_report_with_llm(
                extraction,
                llm,
            )

        case DocumentType.REPAIR_QUOTE:
            return extract_garage_quote_with_llm(
                extraction,
                llm,
            )

        case _:
            raise DocumentExtractionError(
                "Unsupported document type: "
                f"{extraction.asset.type}"
            )
