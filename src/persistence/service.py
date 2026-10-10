"""Application units of work, usable by scripts and future API dependencies."""

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from claims.case import ClaimCase
from claims.models import Claim
from investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.inference import TriagePrediction
from persistence.database import transaction
from persistence.repositories import AssessmentRepository, ClaimNotFoundError, ClaimRepository
from persistence.history import HistoricalClaimRepository
from investigation.history import HistoricalClaimProfile


@dataclass(frozen=True)
class ClaimHistory:
    claim: Claim
    triage: tuple
    anomaly: tuple
    investigation: tuple


class PersistenceService:
    def __init__(self, factory: sessionmaker[Session]):
        self.factory = factory

    def import_claim(self, value: ClaimCase | Claim):
        claim = value.claim if isinstance(value, ClaimCase) else value
        with transaction(self.factory) as session:
            return ClaimRepository(session).add(claim)

    def get_historical_profile(self, claim_id: str) -> HistoricalClaimProfile:
        with self.factory() as session:
            return HistoricalClaimRepository(session).get_profile(claim_id)

    def save_assessments(
        self, claim_id: str, *, triage: TriagePrediction | None = None,
        anomaly: RegisteredAnomalyAssessment | None = None,
        investigation: InvestigationAssessment | None = None,
    ) -> dict:
        """Save already-computed results together, without running inference."""
        if triage is None and anomaly is None and investigation is None:
            raise ValueError("At least one assessment is required")
        with transaction(self.factory) as session:
            repository = AssessmentRepository(session)
            records = {}
            if triage is not None:
                records["triage"] = repository.add_triage(claim_id, triage)
            if anomaly is not None:
                records["anomaly"] = repository.add_anomaly(claim_id, anomaly)
            if investigation is not None:
                records["investigation"] = repository.add_investigation(claim_id, investigation)
            return records

    def get_history(self, claim_id: str) -> ClaimHistory:
        with self.factory() as session:
            claim = ClaimRepository(session).get_by_claim_id(claim_id)
            if claim is None:
                raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
            repository = AssessmentRepository(session)
            return ClaimHistory(
                claim, tuple(repository.list_triage_history(claim_id)),
                tuple(repository.list_anomaly_history(claim_id)),
                tuple(repository.list_investigation_history(claim_id)),
            )
