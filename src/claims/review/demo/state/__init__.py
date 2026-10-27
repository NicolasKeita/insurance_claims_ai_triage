"""Saved component execution state takes precedence over a stale data payload."""

from claims.review.models import ReviewSection


def section(name, value, observation, claim_id, timestamp):
    component = {"documents": "document", "knowledge": "retrieval"}.get(name, name)
    result = observation.get("components", {}).get(component, {})
    status = "AVAILABLE" if value is not None else "NOT_EXECUTED"
    reason = "Previously saved real result; no computation occurred when this review was opened."
    if result.get("status") in {"SKIP", "FAIL"}:
        status = "UNAVAILABLE" if result["status"] == "SKIP" else "FAILED"
        reason = (
            "This component was unavailable in the saved run."
            if status == "UNAVAILABLE"
            else "This component failed in the saved run."
        )
        value = _inventory_only(value) if name == "documents" else None
    elif value is None:
        reason = "No result for this component was saved."
    if name == "similar_claims" and status == "AVAILABLE" and not value.get("results"):
        status, reason = "NO_EVIDENCE", "The saved retrieval completed without matching claims."
    if name == "knowledge" and status == "AVAILABLE" and not value.get("sources"):
        status, reason = "NO_EVIDENCE", "The saved retrieval completed without source evidence."
    return ReviewSection(status=status, data=value, reason=reason, source=f"{name}:{claim_id}", updated_at=timestamp)


def _inventory_only(value):
    if value is None:
        return None
    return {
        "items": [{**item, "extracted_data": None, "extraction_method": None} for item in value["items"]],
        "extractions": {},
    }


def knowledge_state(assessment, observation):
    if assessment.status not in {"AVAILABLE", "NO_EVIDENCE"}:
        return assessment
    generation = observation.get("components", {}).get("rag_generation", {}).get("status")
    if generation in {"SKIP", "FAIL"}:
        assessment.status = "UNAVAILABLE" if generation == "SKIP" else "FAILED"
        assessment.reason = (
            "Source retrieval completed, but answer generation was unavailable or failed in the saved run."
        )
        assessment.data = {**assessment.data, "answer": None, "insufficient_evidence": None}
    elif assessment.data.get("insufficient_evidence"):
        assessment.status = "INSUFFICIENT_EVIDENCE"
    return assessment
