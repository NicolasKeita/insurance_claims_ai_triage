"""Map stored component shapes to the shared presentation schema."""

from claims.investigation.history import HistoricalClaimProfile
from claims.review.stored_context import eligible_similar
from ml.anomaly_training import AnomalyAssessment


def documents_section(observation, claim):
    """Older observations omit ingestion method, which remains unknown."""
    extraction = observation["facts"].get("extraction", {})
    by_kind = {
        kind: extraction[field]
        for kind, field in (("CLAIM_FORM", "form"), ("ACCIDENT_REPORT", "report"), ("REPAIR_QUOTE", "quote"))
        if extraction.get(field) is not None
    }
    items = [
        {
            **entry.model_dump(mode="json"),
            "extraction_method": None,
            "source": "Saved controlled synthetic benchmark documents",
            "extracted_data": by_kind.get(entry.type),
        }
        for entry in claim.documents
    ]
    return {"items": items, "extractions": by_kind}


def anomaly_section(observation):
    value = observation["facts"].get("anomaly")
    if value is None:
        return None
    assessment = AnomalyAssessment.model_validate(value.get("assessment", value)).model_dump(mode="json")
    metadata = observation.get("provenance", {}).get("anomaly", {})
    return {
        "assessment": assessment,
        "model_version": observation.get("model_versions", {}).get("anomaly"),
        "registered_model": metadata.get("checkpoint") or metadata.get("source"),
        "model_alias": "saved-real-result",
    }


def knowledge_section(observation):
    rag = observation.get("rag") or observation["facts"].get("rag")
    if rag is None:
        return None
    answer = rag.get("answer") or {}
    return {
        "answer": answer.get("answer"),
        "insufficient_evidence": answer.get("insufficient_evidence"),
        "sources": rag.get("results", []),
        "citations": answer.get("citations", []),
        "question": rag.get("question"),
    }


def validated_history(value):
    """Reject incomplete stored aggregates rather than creating default zero facts."""
    if value is None:
        return None
    if not isinstance(value, dict) or not set(HistoricalClaimProfile.model_fields).issubset(value):
        raise ValueError("The saved historical profile is incomplete; no counts can be inferred")
    return HistoricalClaimProfile.model_validate(value).model_dump(mode="json")


def filter_similar(value, claim):
    """Retain comparative context only from strictly earlier, different claims."""
    if value is None:
        return None
    return {**value, "results": [item for item in value.get("results", []) if eligible_similar(claim, item)]}


def filter_agent_similar(value, claim):
    """The audit evidence panel follows the same chronology as the comparison table."""
    if value is None:
        return None
    return {
        **value,
        "evidence": [
            item
            for item in value.get("evidence", [])
            if item.get("source_type") != "SIMILAR_CLAIM" or eligible_similar(claim, item.get("structured_data") or {})
        ],
    }
