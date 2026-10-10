"""Versioned rules for review prioritization and signal thresholds."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from investigation.models import InvestigationSignal, ReviewPriority, SignalCode


HISTORICAL_CODES = frozenset({
    SignalCode.RECENT_CLAIM_FREQUENCY, SignalCode.PREVIOUS_ANOMALOUS_CLAIMS,
})
V1_CODES = frozenset(SignalCode) - HISTORICAL_CODES


class InvestigationPolicy(BaseModel):
    """Demo heuristics; weights are policy choices, not insurance standards."""

    model_config = ConfigDict(frozen=True)

    version: str = "investigation_policy_v1"
    signal_weights: Mapping[SignalCode, int] = Field(
        default_factory=lambda: {
            SignalCode.DOCUMENT_FIELD_MISMATCH: 10,
            SignalCode.ADDITIONAL_QUOTE_DAMAGE: 18,
            SignalCode.UNMAPPED_QUOTE_ITEM: 8,
            SignalCode.ANOMALOUS_PATTERN: 18,
            SignalCode.EXTREME_ANOMALY_PERCENTILE: 20,
            SignalCode.MISSING_DOCUMENTS: 12,
        }
    )
    max_review_score: int = Field(default=100, ge=1, le=100)
    extreme_anomaly_percentile: float = Field(default=0.99, ge=0, le=1)
    minimum_document_completeness_ratio: float = Field(default=0.80, ge=0, le=1)
    elevated_priority_min_score: int = Field(default=20, ge=1, le=100)
    high_priority_min_score: int = Field(default=50, ge=1, le=100)
    review_recommendation_min_score: int = Field(default=20, ge=1, le=100)
    # None disables history for v1, preserving its original behavior.
    recent_claims_30_days_threshold: int | None = Field(default=None, ge=1)
    recent_claims_365_days_threshold: int | None = Field(default=None, ge=1)
    previous_anomalous_claims_threshold: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_score_rules(self) -> "InvestigationPolicy":
        history_enabled = any(value is not None for value in (
            self.recent_claims_30_days_threshold,
            self.recent_claims_365_days_threshold,
            self.previous_anomalous_claims_threshold,
        ))
        expected = set(SignalCode) if history_enabled else V1_CODES
        if set(self.signal_weights) != expected:
            raise ValueError("signal_weights must cover exactly the enabled signal codes")
        if self.version == "investigation_policy_v1" and history_enabled:
            raise ValueError("Historical rules require a new policy version")
        if any(weight < 0 for weight in self.signal_weights.values()):
            raise ValueError("signal weights must be nonnegative")
        if not (
            self.elevated_priority_min_score
            < self.high_priority_min_score
            <= self.max_review_score
        ):
            raise ValueError("priority thresholds must increase within score bounds")
        if self.review_recommendation_min_score > self.max_review_score:
            raise ValueError("review recommendation threshold exceeds score maximum")
        # A policy version must not change because a caller edits a nested dict.
        object.__setattr__(
            self, "signal_weights", MappingProxyType(dict(self.signal_weights))
        )
        return self

    @field_serializer("signal_weights")
    def serialize_signal_weights(
        self, weights: Mapping[SignalCode, int]
    ) -> dict[SignalCode, int]:
        return dict(weights)

    def score_signals(self, signals: Iterable[InvestigationSignal]) -> int:
        """Add explicit signal weights and cap the heuristic score."""

        return min(
            self.max_review_score,
            sum(self.signal_weights[signal.code] for signal in signals),
        )

    def priority_for_score(self, score: int) -> ReviewPriority:
        if not 0 <= score <= self.max_review_score:
            raise ValueError("review score is outside policy bounds")
        if score >= self.high_priority_min_score:
            return ReviewPriority.HIGH
        if score >= self.elevated_priority_min_score:
            return ReviewPriority.ELEVATED
        return ReviewPriority.NORMAL

    def review_recommended_for_score(self, score: int) -> bool:
        if not 0 <= score <= self.max_review_score:
            raise ValueError("review score is outside policy bounds")
        return score >= self.review_recommendation_min_score


INVESTIGATION_POLICY_V1 = InvestigationPolicy()
INVESTIGATION_POLICY_V2 = InvestigationPolicy(
    version="investigation_policy_v2",
    signal_weights={
        **INVESTIGATION_POLICY_V1.signal_weights,
        SignalCode.RECENT_CLAIM_FREQUENCY: 15,
        SignalCode.PREVIOUS_ANOMALOUS_CLAIMS: 18,
    },
    recent_claims_30_days_threshold=3,
    recent_claims_365_days_threshold=5,
    previous_anomalous_claims_threshold=2,
)
DEFAULT_INVESTIGATION_POLICY = INVESTIGATION_POLICY_V2
