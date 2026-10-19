"""In-memory retrieval claim repository for local Qdrant orchestration."""

from uuid import NAMESPACE_URL, uuid5

from persistence.repositories import ClaimNotFoundError
from retrieval.models import ClaimRetrievalFacts, StoredRetrievalClaim


class InMemoryRetrievalRepository:
    """Expose controlled claims through the real retrieval repository contract."""

    def __init__(self, claims):
        self.claims = {}
        for claim in claims:
            facts = ClaimRetrievalFacts(
                claim.claim_type.value,
                claim.incident.collision_type.value,
                claim.incident.location,
                claim.vehicle.make,
                claim.vehicle.model,
                claim.vehicle.year,
                tuple(d.value for d in claim.declared_damage),
                claim.incident.injuries_declared,
                claim.repair_estimate.amount,
                claim.repair_estimate.currency,
            )
            item = StoredRetrievalClaim(
                uuid5(NAMESPACE_URL, claim.claim_id), claim.claim_id, claim.incident.date, facts
            )
            self.claims[item.point_id] = item

    def get(self, claim_id):
        for claim in self.claims.values():
            if claim.claim_id == claim_id:
                return claim
        raise ClaimNotFoundError(claim_id)

    def get_many(self, point_ids):
        return {pk: self.claims[pk] for pk in point_ids if pk in self.claims}

    def iter_batches(self, batch_size):
        claims = sorted(self.claims.values(), key=lambda claim: claim.claim_id)
        for start in range(0, len(claims), batch_size):
            yield claims[start : start + batch_size]
