"""Application schema validation for extractions, agent, investigation, vision."""

from pydantic import ValidationError

from agent.models import AgentRunResult
from claims.document_models import AccidentReportExtraction, ClaimFormExtraction, GarageQuoteExtraction
from evaluation.assertions.recording import Check
from investigation.models import InvestigationAssessment
from vision.models import ClaimMultimodalVisionAssessment

DOCUMENT_SCHEMAS = (
    ("form", ClaimFormExtraction),
    ("report", AccidentReportExtraction),
    ("quote", GarageQuoteExtraction),
)
COMPONENT_SCHEMAS = (
    ("agent", AgentRunResult),
    ("investigation", InvestigationAssessment),
    ("vision", ClaimMultimodalVisionAssessment),
)
VISION_COMPUTED = {"detection_count", "observation_summary", "fusion_policy_version"}


def vision_schema_input(value):
    """Remove only documented computed output properties before round-trip QA."""
    if isinstance(value, list):
        return [vision_schema_input(item) for item in value]
    if not isinstance(value, dict):
        return value
    computed = set(VISION_COMPUTED)
    if "damage_assessment" in value and "part_assessment" in value:
        computed.add("filename")
    if "images" in value and "damage_model_metadata" in value:
        computed.update(
            {"observations", "aggregate_mapped_damage_types", "aggregate_vehicle_parts", "ambiguous_associations"}
        )
    return {key: vision_schema_input(item) for key, item in value.items() if key not in computed}


def check_document_schemas(recorder):
    """Verify the three separate document taxonomies and their fields."""
    if recorder.skipped("document"):
        return
    extraction = recorder.observation.facts.get("extraction", {})
    errors = []
    for key, schema in DOCUMENT_SCHEMAS:
        try:
            schema.model_validate(extraction.get(key), extra="forbid")
        except (ValidationError, ValueError) as error:
            errors.append({"document": key, "error": str(error)[:700]})
    recorder.record(
        Check("schema:extraction", "document", "three valid structured document schemas", errors, not errors)
    )


def check_component_schemas(recorder):
    """Verify each structured component against its application schema."""
    observation = recorder.observation
    payloads = {
        "agent": observation.agent,
        "investigation": observation.investigation,
        "vision": observation.vision,
    }
    for component, schema in COMPONENT_SCHEMAS:
        _check_component_schema(recorder, component, payloads[component], schema)


def _check_component_schema(recorder, component, payload, schema):
    if payload is None:
        required = component in {"agent", "investigation"} or recorder.scenario.vision_required
        if required:
            recorder.record(
                Check(
                    f"schema:{component}",
                    component,
                    "valid application schema",
                    None,
                    False,
                    blocked=recorder.skipped(component),
                )
            )
        if component == "agent":
            recorder.record(
                Check(
                    "agent:final_schema",
                    component,
                    "AgentRecommendation",
                    None,
                    False,
                    blocked=recorder.skipped(component),
                )
            )
        return
    try:
        schema.model_validate(vision_schema_input(payload) if component == "vision" else payload, extra="forbid")
        valid, error = True, None
    except (ValidationError, ValueError) as exc:
        valid, error = False, str(exc)[:2000]
    recorder.record(
        Check(f"schema:{component}", component, "valid application schema", "valid" if valid else error, valid)
    )
