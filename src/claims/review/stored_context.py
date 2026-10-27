"""Recover retrieved context from agent evidence without running retrieval again."""

from datetime import date

from claims.review.models import ReviewSection


def context_from_agent(claim, section):
    data = section.data if isinstance(section.data, dict) else {}
    evidence = data.get("evidence", [])
    data["evidence"] = [
        item
        for item in evidence
        if item.get("source_type") != "SIMILAR_CLAIM" or eligible_similar(claim, item.get("structured_data") or {})
    ]
    sources = [_source(item) for item in evidence if item.get("source_type") in {"POLICY_CHUNK", "PROCEDURE_CHUNK"}]
    similar = [item["structured_data"] for item in data["evidence"] if item.get("source_type") == "SIMILAR_CLAIM"]
    common = {"updated_at": section.updated_at} if data.get("run_id") else {}
    return {
        "knowledge": ReviewSection(
            status="AVAILABLE" if sources else "UNAVAILABLE",
            reason=(
                "Retrieved source excerpts from the stored agent run; no standalone RAG answer is persisted."
                if sources
                else "No policy or procedure retrieval snapshot is stored."
            ),
            data={"answer": None, "insufficient_evidence": None, "sources": sources} if sources else None,
            source=f"agent:{data['run_id']}:knowledge" if sources and data.get("run_id") else None,
            **common,
        ),
        "similar_claims": ReviewSection(
            status="AVAILABLE" if similar else "UNAVAILABLE",
            reason="Comparative context from the stored agent run."
            if similar
            else "No similar-claim retrieval snapshot is stored.",
            data={"results": similar} if similar else None,
            source=f"agent:{data['run_id']}:similar_claims" if similar and data.get("run_id") else None,
            **common,
        ),
    }


def _source(item):
    """Preserve source type; similarity may be absent in historical agent evidence."""
    details = item.get("structured_data") or {}
    return {
        **details,
        "source_type": "POLICY" if item["source_type"] == "POLICY_CHUNK" else "PROCEDURE",
        "chunk_id": details.get("chunk_id", item["source_reference"]),
        "title": details.get("title", item["source_reference"]),
        "text": item.get("summary"),
        "similarity_score": details.get("similarity_score"),
    }


def eligible_similar(claim, item):
    """Review context uses the same strictly earlier chronology as repository history."""
    try:
        return item["claim_id"] != claim.claim_id and date.fromisoformat(item["incident_date"]) < claim.incident.date
    except (KeyError, TypeError, ValueError):
        return False
