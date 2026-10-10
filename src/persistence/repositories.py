"""Repositories flush but never commit; the application controls transactions."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from claims.models import Claim
from investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.inference import TriagePrediction
from persistence.mappers import (
    anomaly_from_row, anomaly_to_row, claim_to_row, investigation_from_row,
    investigation_to_row, row_to_claim, triage_from_row, triage_to_row,
)
from persistence.models import (
    AnomalyAssessmentRow, ClaimRow, CustomerRow, InvestigationAssessmentRow,
    PolicyRow, TriageAssessmentRow,
)


class ClaimAlreadyExistsError(ValueError):
    pass


class ClaimNotFoundError(LookupError):
    pass


class PolicyConflictError(ValueError):
    pass


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
        return self.session.scalar(select(ClaimRow.id).where(
            ClaimRow.claim_id == claim_id)) is not None

    def require_pk(self, claim_id: str) -> UUID:
        pk = self.session.scalar(select(ClaimRow.id).where(ClaimRow.claim_id == claim_id))
        if pk is None:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return pk

    def _reuse(self, model, key: str, value: str, **fields):
        # ON CONFLICT DO NOTHING handles concurrent imports without overwriting
        # customer/policy data. PostgreSQL unique constraints remain authoritative.
        pk = self.session.scalar(insert(model).values(**{key: value}, **fields)
                                 .on_conflict_do_nothing(index_elements=[key])
                                 .returning(model.id))
        row = self.session.scalar(select(model).where(getattr(model, key) == value))
        return row, pk is not None

    def add(self, claim: Claim) -> ClaimImportResult:
        if self.exists(claim.claim_id):
            raise ClaimAlreadyExistsError(f"Claim {claim.claim_id} already exists")
        # Build/validate before issuing writes.
        row = claim_to_row(claim, CustomerRow(customer_id=claim.customer.customer_id),
                           PolicyRow(policy_id=claim.policy.policy_id, product=claim.policy.product))
        customer, customer_created = self._reuse(
            CustomerRow, "customer_id", claim.customer.customer_id)
        policy, policy_created = self._reuse(
            PolicyRow, "policy_id", claim.policy.policy_id, product=claim.policy.product)
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
        return ClaimImportResult(claim.claim_id, customer_created, policy_created,
                                 len(row.documents), len(row.images))

    save = add

    def _read(self, statement) -> list[Claim]:
        statement = statement.options(
            selectinload(ClaimRow.customer), selectinload(ClaimRow.policy),
            selectinload(ClaimRow.damages), selectinload(ClaimRow.documents),
            selectinload(ClaimRow.images),
        )
        return [row_to_claim(row) for row in self.session.scalars(statement)]

    def get_by_claim_id(self, claim_id: str) -> Claim | None:
        claims = self._read(select(ClaimRow).where(ClaimRow.claim_id == claim_id))
        return claims[0] if claims else None

    def list_for_customer(self, customer_id: str) -> list[Claim]:
        return self._read(select(ClaimRow).join(ClaimRow.customer).where(
            CustomerRow.customer_id == customer_id).order_by(ClaimRow.incident_date, ClaimRow.claim_id))

    def list_for_policy(self, policy_id: str) -> list[Claim]:
        return self._read(select(ClaimRow).join(ClaimRow.policy).where(
            PolicyRow.policy_id == policy_id).order_by(ClaimRow.incident_date, ClaimRow.claim_id))


class AssessmentRepository:
    """Append/read API only. Anomaly uses (created_at, insertion_order).

    Other histories retain their existing stable UUID tie-break. Anomaly's
    database counter orders timestamp ties by allocation, not commit time.
    """
    def __init__(self, session: Session):
        self.session = session
        self.claims = ClaimRepository(session)

    def _add(self, claim_id, result, to_row, from_row):
        row = to_row(self.claims.require_pk(claim_id), result)
        self.session.add(row)
        self.session.flush()
        return from_row(row)

    def add_triage(self, claim_id: str, result: TriagePrediction):
        return self._add(claim_id, result, triage_to_row, triage_from_row)

    def add_anomaly(self, claim_id: str, result: RegisteredAnomalyAssessment):
        return self._add(claim_id, result, anomaly_to_row, anomaly_from_row)

    def add_investigation(self, claim_id: str, result: InvestigationAssessment):
        return self._add(claim_id, result, investigation_to_row, investigation_from_row)

    def _read(self, claim_id, model, from_row, latest=False):
        pk = self.claims.require_pk(claim_id)
        query = select(model).where(model.claim_pk == pk)
        tie_break = model.insertion_order if model is AnomalyAssessmentRow else model.id
        if latest:
            row = self.session.scalar(query.order_by(model.created_at.desc(), tie_break.desc()).limit(1))
            return from_row(row) if row else None
        return [from_row(row) for row in self.session.scalars(
            query.order_by(model.created_at, tie_break))]

    def list_triage_history(self, claim_id: str):
        return self._read(claim_id, TriageAssessmentRow, triage_from_row)

    def list_anomaly_history(self, claim_id: str):
        return self._read(claim_id, AnomalyAssessmentRow, anomaly_from_row)

    def list_investigation_history(self, claim_id: str):
        return self._read(claim_id, InvestigationAssessmentRow, investigation_from_row)

    def get_latest_triage(self, claim_id: str):
        return self._read(claim_id, TriageAssessmentRow, triage_from_row, latest=True)

    def get_latest_anomaly(self, claim_id: str):
        return self._read(claim_id, AnomalyAssessmentRow, anomaly_from_row, latest=True)

    def get_latest_investigation(self, claim_id: str):
        return self._read(claim_id, InvestigationAssessmentRow, investigation_from_row, latest=True)
