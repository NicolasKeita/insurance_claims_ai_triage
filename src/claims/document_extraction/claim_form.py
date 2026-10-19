"""Deterministic claim-form extraction."""

from claims.document_extraction.base import (
    DocumentExtractionError,
    find_prefixed_value,
    find_section,
    normalize_collision_type,
    normalize_damage,
    parse_document_date,
    parse_yes_no,
    split_vehicle,
    text_lines,
)
from claims.document_models import ClaimFormExtraction, DocumentSource
from claims.enums import DocumentType


def extract_claim_form(extraction) -> ClaimFormExtraction:
    if extraction.asset.type != DocumentType.CLAIM_FORM:
        raise DocumentExtractionError("Expected a claim form document")
    if not extraction.has_text:
        raise DocumentExtractionError("Claim form contains no extractable text")
    lines = text_lines(extraction)
    vehicle_make, vehicle_model = split_vehicle(find_prefixed_value(lines, "Vehicle:"))
    damage_lines = find_section(lines, "Declared damage:", ("Injuries declared:",))
    return ClaimFormExtraction(
        claim_id=find_prefixed_value(lines, "Claim ID:"),
        incident_date=parse_document_date(find_prefixed_value(lines, "Incident date:")),
        location=find_prefixed_value(lines, "Location:"),
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        vehicle_year=int(find_prefixed_value(lines, "Vehicle year:")),
        collision_type=normalize_collision_type(find_prefixed_value(lines, "Collision type:")),
        declared_damage=tuple(normalize_damage(value) for value in damage_lines),
        injuries_declared=parse_yes_no(find_prefixed_value(lines, "Injuries declared:")),
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )
