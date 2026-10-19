"""Translate stored historical facts into review signals."""

from investigation.history import HistoricalClaimProfile
from investigation.models import InvestigationSignal, SignalCategory, SignalCode, SignalSeverity
from investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy


def signals_from_history(
    history: HistoricalClaimProfile,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
) -> tuple[InvestigationSignal, ...]:
    """Translate historical facts once per rule; never consume prior review scores."""
    signals = []
    frequency = _frequency_signal(history, policy)
    if frequency is not None:
        signals.append(frequency)
    previous = _previous_anomaly_signal(history, policy)
    if previous is not None:
        signals.append(previous)
    return tuple(signals)


def _frequency_signal(history, policy):
    windows = (
        (30, history.claims_last_30_days, policy.recent_claims_30_days_threshold),
        (365, history.claims_last_365_days, policy.recent_claims_365_days_threshold),
    )
    triggered = [(days, count) for days, count, threshold in windows if threshold is not None and count >= threshold]
    if not triggered:
        return None
    return InvestigationSignal(
        code=SignalCode.RECENT_CLAIM_FREQUENCY,
        category=SignalCategory.HISTORICAL_CONTEXT,
        severity=SignalSeverity.MEDIUM,
        title="Recent claim frequency",
        explanation=" ".join(
            f"{count} previous claims occurred within the {days} days preceding this incident."
            for days, count in triggered
        ),
        evidence={
            "claims_last_30_days": history.claims_last_30_days,
            "claims_last_365_days": history.claims_last_365_days,
            "previous_claim_count": history.previous_claim_count,
            "policy_30_days_threshold": policy.recent_claims_30_days_threshold,
            "policy_365_days_threshold": policy.recent_claims_365_days_threshold,
        },
    )


def _previous_anomaly_signal(history, policy):
    threshold = policy.previous_anomalous_claims_threshold
    if threshold is None or history.previous_anomalous_claim_count < threshold:
        return None
    return InvestigationSignal(
        code=SignalCode.PREVIOUS_ANOMALOUS_CLAIMS,
        category=SignalCategory.HISTORICAL_CONTEXT,
        severity=SignalSeverity.MEDIUM,
        title="Previous anomaly assessments",
        explanation=(
            f"{history.previous_anomalous_claim_count} previous claims are currently "
            "flagged as anomalous by their latest stored anomaly assessments."
        ),
        evidence={
            "previous_anomalous_claim_count": history.previous_anomalous_claim_count,
            "previous_claim_count": history.previous_claim_count,
            "policy_threshold": threshold,
        },
    )
