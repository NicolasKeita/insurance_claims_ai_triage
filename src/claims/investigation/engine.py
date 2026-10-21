"""Deterministic aggregation of observed facts into review priority."""

from dataclasses import dataclass

from claims.consistency import ClaimConsistencyReport
from claims.investigation.history import HistoricalClaimProfile
from claims.investigation.models import InvestigationAssessment
from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy
from claims.investigation.signals import (
    signals_from_anomaly,
    signals_from_completeness,
    signals_from_consistency,
    signals_from_history,
    signals_from_multimodal,
)
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures
from vision.models import ClaimMultimodalVisionAssessment


@dataclass(frozen=True)
class InvestigationContext:
    """Optional historical and multimodal evidence for one claim assessment."""

    history: HistoricalClaimProfile | None = None
    visual_assessment: ClaimMultimodalVisionAssessment | None = None


def assess_investigation(
    *,
    consistency: ClaimConsistencyReport,
    anomaly: AnomalyAssessment,
    features: TriageFeatures,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
    context: InvestigationContext | None = None,
) -> InvestigationAssessment:
    """Apply a versioned heuristic to evidence; no triage prediction is consumed."""

    history = context.history if context is not None else None
    visual = context.visual_assessment if context is not None else None
    signals = (
        *signals_from_consistency(consistency),
        *signals_from_anomaly(anomaly, policy),
        *signals_from_completeness(features, policy),
        *signals_from_history(history if history is not None else HistoricalClaimProfile(), policy),
        *(signals_from_multimodal(visual, policy) if visual is not None else ()),
    )
    score = policy.score_signals(signals)
    return InvestigationAssessment(
        review_score=score,
        review_priority=policy.priority_for_score(score),
        review_recommended=policy.review_recommended_for_score(score),
        policy_version=policy.version,
        signals=signals,
    )
