"""Claim writes and reads; flushes but never commits."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from claims.models import Claim
from persistence.mappers import claim_to_row, row_to_claim
from persistence.models import ClaimRow, CustomerRow, PolicyRow
from persistence.repositories.errors import ClaimAlreadyExistsError, ClaimNotFoundError, PolicyConflictError


@dataclass(frozen=True)
class ClaimImportResult:
    claim_id: str
    customer_created: bool
    policy_created: bool
    document_count: int
    image_count: int


class ClaimRepository:
    def __init__(self, session: Session):
        self.session = session

    def exists(self, claim_id: str) -> bool:
        return self.session.scalar(select(ClaimRow.id).where(ClaimRow.claim_id == claim_id)) is not None

    def require_pk(self, claim_id: str) -> UUID:
        pk = self.session.scalar(select(ClaimRow.id).where(ClaimRow.claim_id == claim_id))
        if pk is None:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return pk

    def _reuse(self, model, key: str, value: str, **fields):
        """
        ON CONFLICT DO NOTHING handles concurrent imports without overwriting customer/policy data.
        PostgreSQL unique constraints remain authoritative.
        """
        pk = self.session.scalar(
            insert(model)
            .values(**{key: value}, **fields)
            .on_conflict_do_nothing(index_elements=[key])
            .returning(model.id)
        )
        row = self.session.scalar(select(model).where(getattr(model, key) == value))
        return row, pk is not None

    def add(self, claim: Claim) -> ClaimImportResult:
        """
        Build/validate before issuing writes.
        """
        if self.exists(claim.claim_id):
            raise ClaimAlreadyExistsError(f"Claim {claim.claim_id} already exists")
        row = claim_to_row(
            claim,
            CustomerRow(customer_id=claim.customer.customer_id),
            PolicyRow(policy_id=claim.policy.policy_id, product=claim.policy.product),
        )
        customer, customer_created = self._reuse(CustomerRow, "customer_id", claim.customer.customer_id)
        policy, policy_created = self._reuse(
            PolicyRow, "policy_id", claim.policy.policy_id, product=claim.policy.product
        )
        if policy.product != claim.policy.product:
            raise PolicyConflictError(f"Policy {claim.policy.policy_id} has a different product")
        row.customer, row.policy = customer, policy
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError as error:
            if getattr(getattr(error.orig, "diag", None), "constraint_name", None) == "uq_claims_claim_id":
                raise ClaimAlreadyExistsError(f"Claim {claim.claim_id} already exists") from error
            raise
        return ClaimImportResult(claim.claim_id, customer_created, policy_created, len(row.documents), len(row.images))

    save = add

    def _read(self, statement) -> list[Claim]:
        statement = statement.options(
            selectinload(ClaimRow.customer),
            selectinload(ClaimRow.policy),
            selectinload(ClaimRow.damages),
            selectinload(ClaimRow.documents),
            selectinload(ClaimRow.images),
        )
        return [row_to_claim(row) for row in self.session.scalars(statement)]

    def get_by_claim_id(self, claim_id: str) -> Claim | None:
        claims = self._read(select(ClaimRow).where(ClaimRow.claim_id == claim_id))
        return claims[0] if claims else None

    def list_for_customer(self, customer_id: str) -> list[Claim]:
        return self._read(
            select(ClaimRow)
            .join(ClaimRow.customer)
            .where(CustomerRow.customer_id == customer_id)
            .order_by(ClaimRow.incident_date, ClaimRow.claim_id)
        )

    def list_for_policy(self, policy_id: str) -> list[Claim]:
        return self._read(
            select(ClaimRow)
            .join(ClaimRow.policy)
            .where(PolicyRow.policy_id == policy_id)
            .order_by(ClaimRow.incident_date, ClaimRow.claim_id)
        )
