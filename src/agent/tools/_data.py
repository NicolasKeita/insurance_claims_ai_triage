from collections import Counter

from claims.models import Claim

MAX_SIGNALS = 12
MAX_VALUE_CHARS = 800
MAX_KNOWLEDGE_EXCERPT_CHARS = 3200


def _bounded(value, depth: int = 0, budget: list[int] | None = None):
    """
    Bound persisted free text, nested signal facts, and collection fan-out.

    Repository model_dump(mode="json") produces JSON values; unexpected objects fail explicitly rather than hiding
    infrastructure/schema changes.
    """
    budget = [12000] if budget is None else budget
    if budget[0] <= 0:
        return "[data budget exhausted]"
    if isinstance(value, str):
        result = value[: min(MAX_VALUE_CHARS, budget[0])]
        budget[0] -= len(result)
        return result
    if isinstance(value, (bool, int, float)) or value is None:
        budget[0] -= len(str(value))
        return value
    if depth >= 4:
        return "[nested data omitted]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:20]:
            if budget[0] <= 0:
                break
            key = str(key)[:100]
            budget[0] -= len(key)
            result[key] = _bounded(item, depth + 1, budget)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value[:MAX_SIGNALS]:
            if budget[0] <= 0:
                break
            result.append(_bounded(item, depth + 1, budget))
        return result
    raise TypeError("Agent tool data must be JSON-compatible")


def _claim_data(claim: Claim) -> dict:
    documents = Counter((d.type.value, d.available) for d in claim.documents)
    return {
        "claim_id": claim.claim_id,
        "claim_type": claim.claim_type.value,
        "status": claim.status.value,
        "policy_product": claim.policy.product[:MAX_VALUE_CHARS],
        "incident": _bounded(claim.incident.model_dump(mode="json")),
        "vehicle": _bounded(claim.vehicle.model_dump(mode="json")),
        "repair_estimate": claim.repair_estimate.model_dump(mode="json"),
        "declared_damage": sorted({damage.value for damage in claim.declared_damage}),
        "documents": [
            {"type": kind, "available": available, "count": count}
            for (kind, available), count in sorted(documents.items())
        ],
        "document_count": len(claim.documents),
        "image_count": len(claim.images),
    }
