"""Deterministic accident-report extraction."""

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
from claims.document_models import AccidentReportExtraction, DocumentSource
from claims.enums import DocumentType


def extract_accident_report(extraction) -> AccidentReportExtraction:
    if extraction.asset.type != DocumentType.ACCIDENT_REPORT:
        raise DocumentExtractionError("Expected an accident report document")
    if not extraction.has_text:
        raise DocumentExtractionError("Accident report contains no extractable text")
    lines = text_lines(extraction)
    vehicle_make, vehicle_model = split_vehicle(_vehicle_line(lines))
    damage_lines = find_section(lines, "Reported damage:", ("Passengers:", "Injuries:"))
    return AccidentReportExtraction(
        claim_id=find_prefixed_value(lines, "Claim ID:"),
        incident_date=parse_document_date(find_prefixed_value(lines, "Date:")),
        location=find_prefixed_value(lines, "Location:"),
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        collision_type=normalize_collision_type(find_prefixed_value(lines, "Collision type:")),
        reported_damage=tuple(normalize_damage(value) for value in damage_lines),
        passengers=int(find_prefixed_value(lines, "Passengers:")),
        injuries_declared=parse_yes_no(find_prefixed_value(lines, "Injuries:")),
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )


def _vehicle_line(lines):
    try:
        vehicle_index = lines.index("Vehicle:")
        return lines[vehicle_index + 1]
    except (ValueError, IndexError) as error:
        raise DocumentExtractionError("Missing vehicle information") from error
