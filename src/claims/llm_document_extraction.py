from pydantic import ValidationError

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
Extract the repair quote information from the document below.

Rules:
- Use only information explicitly present in the document.
- Do not invent missing values.
- If garage, claim ID, labor or total is missing, return null.
- Parts must contain repair/replacement line items.
- Do not include Labor in parts.
- Do not include Total in parts.
- Preserve monetary values exactly as written.
- Preserve the three-letter currency code.

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
    try:
        return GarageQuoteExtraction(
            garage=result.garage,
            claim_id=result.claim_id,
            parts=result.parts,
            labor=result.labor,
            total=result.total,
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