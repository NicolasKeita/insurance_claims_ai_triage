"""Auditable, JSON-compatible outputs of deterministic investigation review rules."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class SignalSeverity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SignalCategory(StrEnum):
    DOCUMENT_CONSISTENCY = "DOCUMENT_CONSISTENCY"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    DATA_COMPLETENESS = "DATA_COMPLETENESS"
    HISTORICAL_CONTEXT = "HISTORICAL_CONTEXT"


class SignalCode(StrEnum):
    DOCUMENT_FIELD_MISMATCH = "DOCUMENT_FIELD_MISMATCH"
    ADDITIONAL_QUOTE_DAMAGE = "ADDITIONAL_QUOTE_DAMAGE"
    UNMAPPED_QUOTE_ITEM = "UNMAPPED_QUOTE_ITEM"
    ANOMALOUS_PATTERN = "ANOMALOUS_PATTERN"
    EXTREME_ANOMALY_PERCENTILE = "EXTREME_ANOMALY_PERCENTILE"
    MISSING_DOCUMENTS = "MISSING_DOCUMENTS"
    RECENT_CLAIM_FREQUENCY = "RECENT_CLAIM_FREQUENCY"
    PREVIOUS_ANOMALOUS_CLAIMS = "PREVIOUS_ANOMALOUS_CLAIMS"


class ReviewPriority(StrEnum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


class InvestigationSignal(BaseModel):
    """An observed reason for human review, with serializable supporting facts."""

    model_config = ConfigDict(frozen=True)

    code: SignalCode
    category: SignalCategory
    severity: SignalSeverity
    title: str
    explanation: str
    evidence: dict[str, JsonValue]


class InvestigationAssessment(BaseModel):
    """A versioned heuristic priority assessment, never a probability."""

    model_config = ConfigDict(frozen=True)

    review_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Deterministic heuristic review-prioritization score defined by the "
            "policy; not a statistical probability."
        ),
    )
    review_priority: ReviewPriority
    review_recommended: bool
    policy_version: str
    signals: tuple[InvestigationSignal, ...]
