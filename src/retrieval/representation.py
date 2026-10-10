"""Stable semantic facts only: no identity, date, workflow or AI outputs."""

from retrieval.models import ClaimRetrievalDocument, ClaimRetrievalFacts, StoredRetrievalClaim

CLAIM_RETRIEVAL_REPRESENTATION_VERSION = "claim_retrieval_v1"


def _text(value: str) -> str:
    return " ".join(value.split())


def build_claim_retrieval_text(facts: ClaimRetrievalFacts) -> str:
    # Fixed-point formatting avoids scientific notation and DB scale differences.
    amount = format(facts.repair_amount, "f")
    if "." in amount:
        amount = amount.rstrip("0").rstrip(".")
    fields = (
        ("claim_type", facts.claim_type),
        ("collision_type", facts.collision_type),
        ("location", facts.location),
        ("vehicle_make", facts.vehicle_make),
        ("vehicle_model", facts.vehicle_model),
        ("vehicle_year", str(facts.vehicle_year)),
        ("declared_damage", ", ".join(sorted(set(facts.declared_damage))) or "none"),
        ("injuries_declared", "true" if facts.injuries_declared else "false"),
        ("repair_estimate_amount", amount),
        ("repair_estimate_currency", facts.repair_currency),
    )
    return "\n".join(f"{label}: {_text(value)}" for label, value in fields)


def build_retrieval_document(claim: StoredRetrievalClaim) -> ClaimRetrievalDocument:
    return ClaimRetrievalDocument(
        point_id=claim.point_id, claim_id=claim.claim_id,
        incident_date=claim.incident_date, claim_type=claim.facts.claim_type,
        canonical_text=build_claim_retrieval_text(claim.facts),
        representation_version=CLAIM_RETRIEVAL_REPRESENTATION_VERSION,
    )
