"""Deterministic aggregation of observed facts into review priority."""

from claims.consistency import ClaimConsistencyReport
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures

from investigation.models import InvestigationAssessment
from investigation.history import HistoricalClaimProfile
from investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy
from investigation.signals import (
    signals_from_anomaly,
    signals_from_completeness,
    signals_from_consistency,
    signals_from_history,
)


def assess_investigation(
    *,
    consistency: ClaimConsistencyReport,
    anomaly: AnomalyAssessment,
    features: TriageFeatures,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
    history: HistoricalClaimProfile | None = None,
) -> InvestigationAssessment:
    """Apply a versioned heuristic to evidence; no triage prediction is consumed."""

    signals = (
        *signals_from_consistency(consistency),
        *signals_from_anomaly(anomaly, policy),
        *signals_from_completeness(features, policy),
        *signals_from_history(history if history is not None else HistoricalClaimProfile(), policy),
    )
    score = policy.score_signals(signals)
    return InvestigationAssessment(
        review_score=score,
        review_priority=policy.priority_for_score(score),
        review_recommended=policy.review_recommended_for_score(score),
        policy_version=policy.version,
        signals=signals,
    )
