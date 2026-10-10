"""Database-backed investigation orchestration, with domain inputs only."""

from collections.abc import Callable

from sqlalchemy.orm import Session

from claims.consistency import ClaimConsistencyReport
from claims.models import Claim
from investigation.engine import assess_investigation
from investigation.policy import INVESTIGATION_POLICY_V2
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.features import TriageFeatures, calculate_vehicle_age, count_consistency_issues
from persistence.history import HistoricalClaimRepository
from persistence.repositories import AssessmentRepository, ClaimNotFoundError, ClaimRepository


def features_from_stored_claim(claim: Claim, consistency: ClaimConsistencyReport) -> TriageFeatures:
    """Use persisted estimate/metadata; document comparison is caller-supplied.

    Extracted consistency reports are not currently persisted. Do not fabricate
    document matches or require a filesystem ClaimCase for a stored claim.
    """
    count = len(claim.documents)
    missing = sum(not document.available for document in claim.documents)
    return TriageFeatures(
        claim_type=claim.claim_type, collision_type=claim.incident.collision_type,
        vehicle_age=calculate_vehicle_age(claim.vehicle.year, claim.incident.date.year),
        repair_amount=float(claim.repair_estimate.amount),
        declared_damage_count=len(claim.declared_damage),
        additional_damage_count=len(consistency.additional_quote_damage),
        document_count=count, missing_document_count=missing,
        document_completeness_ratio=(count - missing) / count if count else 1.0,
        injuries_declared=claim.incident.injuries_declared,
        consistency_issue_count=count_consistency_issues(consistency),
    )


def assess_stored_claim(
    session: Session, claim_id: str, consistency: ClaimConsistencyReport,
    anomaly_assessor: Callable[[TriageFeatures], RegisteredAnomalyAssessment],
):
    """Append current anomaly and v2 investigation atomically; caller commits."""
    claim = ClaimRepository(session).get_by_claim_id(claim_id)
    if claim is None:
        raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
    history = HistoricalClaimRepository(session).get_profile(claim_id)
    features = features_from_stored_claim(claim, consistency)
    anomaly = anomaly_assessor(features)
    result = assess_investigation(
        consistency=consistency, anomaly=anomaly.assessment, features=features,
        history=history, policy=INVESTIGATION_POLICY_V2,
    )
    repository = AssessmentRepository(session)
    repository.add_anomaly(claim_id, anomaly)
    return repository.add_investigation(claim_id, result).result
