"""Request and response models for the assessment endpoints."""

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from claims.consistency import ClaimConsistencyReport
from investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures, count_consistency_issues
from ml.inference import TriagePrediction


class ClaimsAssessment(BaseModel):
    triage: TriagePrediction
    anomaly: RegisteredAnomalyAssessment
    investigation: InvestigationAssessment


class StructuredEvidenceRequest(BaseModel):
    features: TriageFeatures
    consistency: ClaimConsistencyReport

    @model_validator(mode="after")
    def validate_consistency_counts(self) -> Self:
        if self.features.additional_damage_count != len(self.consistency.additional_quote_damage):
            raise ValueError("additional_damage_count must match additional_quote_damage")
        if self.features.consistency_issue_count != count_consistency_issues(self.consistency):
            raise ValueError("consistency_issue_count must match consistency report")
        return self


class InvestigationAssessmentRequest(StructuredEvidenceRequest):
    anomaly: AnomalyAssessment


class ClaimsAssessmentRequest(StructuredEvidenceRequest):
    pass


class StoredInvestigationRequest(BaseModel):
    # Only comparisons are supplied: stored claims and history are authoritative.
    model_config = ConfigDict(extra="forbid")
    consistency: ClaimConsistencyReport
