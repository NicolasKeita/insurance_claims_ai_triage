"""Explicit, service-free conversion at the domain/storage boundary."""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

from claims.models import Claim
from investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.inference import TriagePrediction
from persistence.models import (
    AnomalyAssessmentRow, ClaimRow, CustomerRow, DeclaredDamageRow, DocumentRow,
    ImageRow, InvestigationAssessmentRow, PolicyRow, TriageAssessmentRow,
)

Result = TypeVar("Result", bound=BaseModel)


@dataclass(frozen=True)
class AssessmentRecord(Generic[Result]):
    id: UUID
    created_at: datetime
    result: Result


def claim_to_row(claim: Claim, customer: CustomerRow, policy: PolicyRow) -> ClaimRow:
    # Validate again: model_copy/model_construct can bypass Pydantic validation.
    claim = Claim.model_validate(claim.model_dump())
    if len(set(claim.declared_damage)) != len(claim.declared_damage):
        raise ValueError("A declared damage type may occur only once per claim")
    return ClaimRow(
        claim_id=claim.claim_id, customer=customer, policy=policy,
        claim_type=claim.claim_type, status=claim.status,
        incident_date=claim.incident.date, incident_location=claim.incident.location,
        collision_type=claim.incident.collision_type,
        injuries_declared=claim.incident.injuries_declared,
        vehicle_make=claim.vehicle.make, vehicle_model=claim.vehicle.model,
        vehicle_year=claim.vehicle.year,
        repair_estimate_amount=claim.repair_estimate.amount,
        repair_estimate_currency=claim.repair_estimate.currency,
        damages=[DeclaredDamageRow(damage_type=d, position=i)
                 for i, d in enumerate(claim.declared_damage)],
        documents=[DocumentRow(document_type=d.type, filename=d.filename,
                               available=d.available, position=i)
                   for i, d in enumerate(claim.documents)],
        images=[ImageRow(filename=f, position=i) for i, f in enumerate(claim.images)],
    )


def row_to_claim(row: ClaimRow) -> Claim:
    return Claim.model_validate({
        "claim_id": row.claim_id, "claim_type": row.claim_type, "status": row.status,
        "customer": {"customer_id": row.customer.customer_id},
        "policy": {"policy_id": row.policy.policy_id, "product": row.policy.product},
        "incident": {"date": row.incident_date, "location": row.incident_location,
                     "collision_type": row.collision_type,
                     "injuries_declared": row.injuries_declared},
        "vehicle": {"make": row.vehicle_make, "model": row.vehicle_model,
                    "year": row.vehicle_year},
        "repair_estimate": {"amount": row.repair_estimate_amount,
                            "currency": row.repair_estimate_currency},
        "declared_damage": [d.damage_type for d in row.damages],
        "documents": [{"type": d.document_type, "filename": d.filename,
                       "available": d.available} for d in row.documents],
        "images": [image.filename for image in row.images],
    })


def checked_triage(result: TriagePrediction) -> TriagePrediction:
    result = TriagePrediction.model_validate(result.model_dump())
    probabilities = result.probabilities
    if (not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1
                                 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
            or result.recommended_workflow.value not in probabilities
            or not math.isclose(result.confidence,
                                probabilities[result.recommended_workflow.value],
                                abs_tol=1e-6)):
        raise ValueError("Incoherent triage probabilities/confidence")
    return result


def triage_to_row(claim_pk: UUID, result: TriagePrediction) -> TriageAssessmentRow:
    result = checked_triage(result)
    return TriageAssessmentRow(claim_pk=claim_pk, **result.model_dump(mode="json"))


def anomaly_to_row(claim_pk: UUID, result: RegisteredAnomalyAssessment) -> AnomalyAssessmentRow:
    result = RegisteredAnomalyAssessment.model_validate(result.model_dump())
    assessment = result.assessment
    if not all(math.isfinite(v) for v in (
        assessment.anomaly_score, assessment.threshold, assessment.reference_percentile
    )):
        raise ValueError("Anomaly numbers must be finite")
    return AnomalyAssessmentRow(
        claim_pk=claim_pk, is_anomalous=assessment.is_anomalous,
        anomaly_score=assessment.anomaly_score, threshold=assessment.threshold,
        reference_percentile=assessment.reference_percentile,
        unusual_features=list(assessment.statistical_signals),
        registered_model=result.registered_model, model_version=result.model_version,
        model_alias=result.model_alias,
    )


def investigation_to_row(claim_pk: UUID, result: InvestigationAssessment) -> InvestigationAssessmentRow:
    result = InvestigationAssessment.model_validate(result.model_dump())
    payload = result.model_dump(mode="json")
    # A JSON round-trip copies nested evidence, and rejects NaN/Infinity.
    payload["signals"] = json.loads(json.dumps(payload["signals"], allow_nan=False))
    return InvestigationAssessmentRow(claim_pk=claim_pk, **payload)


def triage_from_row(row: TriageAssessmentRow) -> AssessmentRecord[TriagePrediction]:
    result = checked_triage(TriagePrediction(
        recommended_workflow=row.recommended_workflow, confidence=row.confidence,
        probabilities=row.probabilities, registered_model=row.registered_model,
        model_version=row.model_version, model_alias=row.model_alias,
    ))
    return AssessmentRecord(row.id, row.created_at, result)


def anomaly_from_row(row: AnomalyAssessmentRow) -> AssessmentRecord[RegisteredAnomalyAssessment]:
    result = RegisteredAnomalyAssessment.model_validate({
        "registered_model": row.registered_model, "model_version": row.model_version,
        "model_alias": row.model_alias,
        "assessment": {"is_anomalous": row.is_anomalous,
                       "anomaly_score": row.anomaly_score, "threshold": row.threshold,
                       "reference_percentile": row.reference_percentile,
                       "statistical_signals": row.unusual_features},
    })
    return AssessmentRecord(row.id, row.created_at, result)


def investigation_from_row(row: InvestigationAssessmentRow) -> AssessmentRecord[InvestigationAssessment]:
    result = InvestigationAssessment(
        review_score=row.review_score, review_priority=row.review_priority,
        review_recommended=row.review_recommended, policy_version=row.policy_version,
        signals=row.signals,
    )
    return AssessmentRecord(row.id, row.created_at, result)
