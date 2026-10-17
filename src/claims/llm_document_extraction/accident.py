"""LLM-assisted accident-report extraction."""

from claims.document_extraction import DocumentExtractionError
from claims.document_models import AccidentReportExtraction, AccidentReportLlmOutput, DocumentSource
from claims.enums import DocumentType
from claims.llm import StructuredLlm
from claims.llm_document_extraction.common import parse_llm_date
from claims.llm_document_extraction.prompts import build_accident_report_prompt


def extract_accident_report_with_llm(extraction, llm: StructuredLlm) -> AccidentReportExtraction:
    if extraction.asset.type != DocumentType.ACCIDENT_REPORT:
        raise DocumentExtractionError("Expected an accident report document")
    if not extraction.has_text:
        raise DocumentExtractionError("Accident report contains no extractable text")
    result = llm.generate(prompt=build_accident_report_prompt(extraction), response_model=AccidentReportLlmOutput)
    missing_fields = _missing_accident_fields(result)
    if missing_fields:
        raise DocumentExtractionError("Missing required accident report fields: " + ", ".join(missing_fields))
    return AccidentReportExtraction(
        claim_id=result.claim_id,
        incident_date=parse_llm_date(result.incident_date),
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


def _missing_accident_fields(result):
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
    return [field for field, value in required_fields.items() if value is None]
