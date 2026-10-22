"""Representative preflight rows; no services or evaluation pipeline involved."""

from evaluation.preflight import PreflightReport


def missing_services(status="SKIP"):
    reasons = {
        "postgresql": "RuntimeError: DATABASE_URL is not configured; no test-database fallback is attempted",
        "qdrant": "ResponseHandlingException: timed out",
        "knowledge_collections": "RuntimeError: Qdrant is unavailable",
        "similar_claims_collection": "RuntimeError: Qdrant is unavailable",
        "stored_claim:controlled": "RuntimeError: PostgreSQL is unavailable",
    }
    rows = [{"component": name, "status": status, "reason": reason} for name, reason in reasons.items()]
    rows.append({"component": "prerequisites:controlled", "status": status, "missing_components": list(reasons)})
    return PreflightReport(
        rows=rows, blocked_scenarios={"controlled": [f"{key}: {value}" for key, value in reasons.items()]}
    )
