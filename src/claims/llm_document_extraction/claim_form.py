"""LLM-assisted claim-form extraction."""

from claims.document_extraction import DocumentExtractionError
from claims.document_models import ClaimFormExtraction, ClaimFormLlmOutput, DocumentSource
from claims.enums import DocumentType
from claims.llm import StructuredLlm
from claims.llm_document_extraction.common import parse_llm_date
from claims.llm_document_extraction.prompts import build_claim_form_prompt


def extract_claim_form_with_llm(extraction, llm: StructuredLlm) -> ClaimFormExtraction:
    if extraction.asset.type != DocumentType.CLAIM_FORM:
        raise DocumentExtractionError("Expected a claim form document")
    if not extraction.has_text:
        raise DocumentExtractionError("Claim form contains no extractable text")
    result = llm.generate(prompt=build_claim_form_prompt(extraction), response_model=ClaimFormLlmOutput)
    missing_fields = _missing_claim_form_fields(result)
    if missing_fields:
        raise DocumentExtractionError("Missing required claim form fields: " + ", ".join(missing_fields))
    return ClaimFormExtraction(
        claim_id=result.claim_id,
        incident_date=parse_llm_date(result.incident_date),
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


def _missing_claim_form_fields(result):
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
    return [field for field, value in required_fields.items() if value is None]
