"""Assessment <-> row conversion with finite-number and JSON validation."""

import json
import math
from uuid import UUID

from claims.investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.inference import TriagePrediction
from persistence.mappers.common import AssessmentRecord
from persistence.models import AnomalyAssessmentRow, InvestigationAssessmentRow, TriageAssessmentRow


def checked_triage(result: TriagePrediction) -> TriagePrediction:
    result = TriagePrediction.model_validate(result.model_dump())
    probabilities = result.probabilities
    if (
        not probabilities
        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
        or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
        or result.recommended_workflow.value not in probabilities
        or not math.isclose(result.confidence, probabilities[result.recommended_workflow.value], abs_tol=1e-6)
    ):
        raise ValueError("Incoherent triage probabilities/confidence")
    return result


def triage_to_row(claim_pk: UUID, result: TriagePrediction) -> TriageAssessmentRow:
    result = checked_triage(result)
    return TriageAssessmentRow(claim_pk=claim_pk, **result.model_dump(mode="json"))


def anomaly_to_row(claim_pk: UUID, result: RegisteredAnomalyAssessment) -> AnomalyAssessmentRow:
    result = RegisteredAnomalyAssessment.model_validate(result.model_dump())
    assessment = result.assessment
    if not all(
        math.isfinite(v) for v in (assessment.anomaly_score, assessment.threshold, assessment.reference_percentile)
    ):
        raise ValueError("Anomaly numbers must be finite")
    return AnomalyAssessmentRow(
        claim_pk=claim_pk,
        is_anomalous=assessment.is_anomalous,
        anomaly_score=assessment.anomaly_score,
        threshold=assessment.threshold,
        reference_percentile=assessment.reference_percentile,
        unusual_features=list(assessment.statistical_signals),
        registered_model=result.registered_model,
        model_version=result.model_version,
        model_alias=result.model_alias,
    )


def investigation_to_row(claim_pk: UUID, result: InvestigationAssessment) -> InvestigationAssessmentRow:
    """
    A JSON round-trip copies nested evidence, and rejects NaN/Infinity.
    """
    result = InvestigationAssessment.model_validate(result.model_dump())
    payload = result.model_dump(mode="json")
    payload["signals"] = json.loads(json.dumps(payload["signals"], allow_nan=False))
    return InvestigationAssessmentRow(claim_pk=claim_pk, **payload)


def triage_from_row(row: TriageAssessmentRow) -> AssessmentRecord[TriagePrediction]:
    result = checked_triage(
        TriagePrediction(
            recommended_workflow=row.recommended_workflow,
            confidence=row.confidence,
            probabilities=row.probabilities,
            registered_model=row.registered_model,
            model_version=row.model_version,
            model_alias=row.model_alias,
        )
    )
    return AssessmentRecord(row.id, row.created_at, result)


def anomaly_from_row(row: AnomalyAssessmentRow) -> AssessmentRecord[RegisteredAnomalyAssessment]:
    result = RegisteredAnomalyAssessment.model_validate(
        {
            "registered_model": row.registered_model,
            "model_version": row.model_version,
            "model_alias": row.model_alias,
            "assessment": {
                "is_anomalous": row.is_anomalous,
                "anomaly_score": row.anomaly_score,
                "threshold": row.threshold,
                "reference_percentile": row.reference_percentile,
                "statistical_signals": row.unusual_features,
            },
        }
    )
    return AssessmentRecord(row.id, row.created_at, result)


def investigation_from_row(row: InvestigationAssessmentRow) -> AssessmentRecord[InvestigationAssessment]:
    result = InvestigationAssessment(
        review_score=row.review_score,
        review_priority=row.review_priority,
        review_recommended=row.review_recommended,
        policy_version=row.policy_version,
        signals=row.signals,
    )
    return AssessmentRecord(row.id, row.created_at, result)
