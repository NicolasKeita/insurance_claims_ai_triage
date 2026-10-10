"""Deterministic test vectors; no semantic quality claims, model or network."""

from datetime import date
from decimal import Decimal
from uuid import UUID

import numpy as np

from persistence.repositories import ClaimNotFoundError
from retrieval.models import ClaimRetrievalFacts, StoredRetrievalClaim


class FakeEmbeddings:
    dimension = 4
    model_name = "deterministic-test-only-v1"
    normalized = True

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            fields = dict(line.split(": ", 1) for line in text.splitlines())
            collision = fields["collision_type"]
            vector = np.array([
                float(collision == "FRONT_COLLISION"),
                float(collision == "PARKING_DAMAGE"),
                float(collision == "SIDE_COLLISION"),
                float(fields["repair_estimate_amount"]) / 10000 + 0.1,
            ])
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def claim(number=1, *, days=0, amount="3160", collision="FRONT_COLLISION", claim_type="AUTO_COLLISION"):
    from datetime import timedelta

    return StoredRetrievalClaim(
        UUID(int=number), f"DEMO-TEST-{number}", date(2026, 9, 20) - timedelta(days=days),
        ClaimRetrievalFacts(claim_type, collision, "Bordeaux", "Renault", "Clio", 2021,
                            ("LEFT_HEADLIGHT", "FRONT_BUMPER"), False, Decimal(amount), "EUR"),
    )


class FakeRepository:
    def __init__(self, claims):
        self.claims = {c.point_id: c for c in claims}
        self.bulk_reads = []

    def get(self, claim_id):
        for item in self.claims.values():
            if item.claim_id == claim_id:
                return item
        raise ClaimNotFoundError(f"Claim {claim_id} does not exist")

    def get_many(self, point_ids):
        self.bulk_reads.append(point_ids)
        return {pk: self.claims[pk] for pk in point_ids if pk in self.claims}

    def iter_batches(self, batch_size):
        ordered = sorted(self.claims.values(), key=lambda c: c.point_id)
        for start in range(0, len(ordered), batch_size):
            yield ordered[start:start + batch_size]
