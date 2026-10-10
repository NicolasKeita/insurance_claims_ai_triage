"""Batch reads into retrieval DTOs. No ORM object leaves this boundary."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import load_only, selectinload

from persistence.models import ClaimRow
from persistence.repositories import ClaimNotFoundError
from retrieval.models import ClaimRetrievalFacts, StoredRetrievalClaim


def _map(row: ClaimRow) -> StoredRetrievalClaim:
    return StoredRetrievalClaim(
        point_id=row.id, claim_id=row.claim_id, incident_date=row.incident_date,
        facts=ClaimRetrievalFacts(
            claim_type=row.claim_type.value, collision_type=row.collision_type.value,
            location=row.incident_location, vehicle_make=row.vehicle_make,
            vehicle_model=row.vehicle_model, vehicle_year=row.vehicle_year,
            declared_damage=tuple(d.damage_type.value for d in row.damages),
            injuries_declared=row.injuries_declared,
            repair_amount=row.repair_estimate_amount,
            repair_currency=row.repair_estimate_currency,
        ),
    )


class RetrievalClaimRepository:
    def __init__(self, factory):
        self.factory = factory

    def _read(self, statement):
        # Two queries per batch: facts and damages. No customer, policy or outputs.
        with self.factory() as session:
            rows = session.scalars(statement.options(
                load_only(
                    ClaimRow.id, ClaimRow.claim_id, ClaimRow.incident_date,
                    ClaimRow.claim_type, ClaimRow.collision_type, ClaimRow.incident_location,
                    ClaimRow.vehicle_make, ClaimRow.vehicle_model, ClaimRow.vehicle_year,
                    ClaimRow.injuries_declared, ClaimRow.repair_estimate_amount,
                    ClaimRow.repair_estimate_currency, raiseload=True,
                ),
                selectinload(ClaimRow.damages),
            ))
            return [_map(row) for row in rows]

    def get(self, claim_id: str) -> StoredRetrievalClaim:
        values = self._read(select(ClaimRow).where(ClaimRow.claim_id == claim_id))
        if not values:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return values[0]

    def get_many(self, point_ids: list[UUID]) -> dict[UUID, StoredRetrievalClaim]:
        if not point_ids:
            return {}
        return {c.point_id: c for c in self._read(select(ClaimRow).where(ClaimRow.id.in_(point_ids)))}

    def iter_batches(self, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        after = None
        while True:
            statement = select(ClaimRow).order_by(ClaimRow.id).limit(batch_size)
            if after is not None:
                statement = statement.where(ClaimRow.id > after)
            batch = self._read(statement)
            if not batch:
                break
            # Session is closed before inference or Qdrant calls.
            yield batch
            after = batch[-1].point_id
