"""Database-backed investigation orchestration, with domain inputs only."""

from collections.abc import Callable

from sqlalchemy.orm import Session

from claims.consistency import ClaimConsistencyReport
from claims.investigation.engine import InvestigationContext, assess_investigation
from claims.investigation.policy import INVESTIGATION_POLICY_V2, INVESTIGATION_POLICY_V3
from claims.models import Claim
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.features import TriageFeatures, calculate_vehicle_age, count_consistency_issues
from persistence.history import HistoricalClaimRepository
from persistence.repositories import AssessmentRepository, ClaimNotFoundError, ClaimRepository
from persistence.vision_assessments import VisionAssessmentRepository
from vision.models import ClaimMultimodalVisionAssessment


def features_from_stored_claim(claim: Claim, consistency: ClaimConsistencyReport) -> TriageFeatures:
    """Use persisted estimate/metadata; document comparison is caller-supplied.

    Extracted consistency reports are not currently persisted. Do not fabricate
    document matches or require a filesystem ClaimCase for a stored claim.
    """
    count = len(claim.documents)
    missing = sum(not document.available for document in claim.documents)
    return TriageFeatures(
        claim_type=claim.claim_type,
        collision_type=claim.incident.collision_type,
        vehicle_age=calculate_vehicle_age(claim.vehicle.year, claim.incident.date.year),
        repair_amount=float(claim.repair_estimate.amount),
        declared_damage_count=len(claim.declared_damage),
        additional_damage_count=len(consistency.additional_quote_damage),
        document_count=count,
        missing_document_count=missing,
        document_completeness_ratio=(count - missing) / count if count else 1.0,
        injuries_declared=claim.incident.injuries_declared,
        consistency_issue_count=count_consistency_issues(consistency),
    )


def assess_stored_claim(
    session: Session,
    claim_id: str,
    consistency: ClaimConsistencyReport,
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
        consistency=consistency,
        anomaly=anomaly.assessment,
        features=features,
        policy=INVESTIGATION_POLICY_V2,
        context=InvestigationContext(history=history),
    )
    repository = AssessmentRepository(session)
    repository.add_anomaly(claim_id, anomaly)
    return repository.add_investigation(claim_id, result).result


def assess_stored_claim_multimodal(
    session: Session,
    claim_id: str,
    consistency: ClaimConsistencyReport,
    anomaly_assessor: Callable[[TriageFeatures], RegisteredAnomalyAssessment],
    visual_assessment: ClaimMultimodalVisionAssessment,
):
    """Append computed vision, anomaly and v3 investigation atomically.

    The caller commits. Vision inference is completed outside this SQL unit of
    work; neither this service nor an agent initiates it implicitly.
    """
    claim = ClaimRepository(session).get_by_claim_id(claim_id)
    if claim is None:
        raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
    if visual_assessment.claim_id != claim_id:
        raise ValueError("Visual assessment belongs to a different claim")
    if set(visual_assessment.consistency.declared_damage_types) != set(claim.declared_damage):
        raise ValueError("Visual assessment declaration snapshot differs from the stored claim")
    history = HistoricalClaimRepository(session).get_profile(claim_id)
    features = features_from_stored_claim(claim, consistency)
    anomaly = anomaly_assessor(features)
    result = assess_investigation(
        consistency=consistency,
        anomaly=anomaly.assessment,
        features=features,
        policy=INVESTIGATION_POLICY_V3,
        context=InvestigationContext(history=history, visual_assessment=visual_assessment),
    )
    VisionAssessmentRepository(session).add(visual_assessment)
    repository = AssessmentRepository(session)
    repository.add_anomaly(claim_id, anomaly)
    return repository.add_investigation(claim_id, result).result
