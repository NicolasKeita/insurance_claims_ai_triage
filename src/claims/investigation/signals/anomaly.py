"""Translate anomaly assessments into descriptive review signals."""

from claims.investigation.models import InvestigationSignal, SignalCategory, SignalCode, SignalSeverity
from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy
from ml.anomaly_training import AnomalyAssessment


def signals_from_anomaly(
    anomaly: AnomalyAssessment,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
) -> tuple[InvestigationSignal, ...]:
    """Report a model flag and independent rarity descriptors, without attribution claims."""

    signals: list[InvestigationSignal] = []
    flagged = _flagged_signal(anomaly)
    if flagged is not None:
        signals.append(flagged)
    extreme = _extreme_signal(anomaly, policy)
    if extreme is not None:
        signals.append(extreme)
    return tuple(signals)


def _flagged_signal(anomaly):
    if not anomaly.is_anomalous:
        return None
    evidence = {
        "anomaly_score": anomaly.anomaly_score,
        "threshold": anomaly.threshold,
        "reference_percentile": anomaly.reference_percentile,
        "unusual_features": sorted(set(anomaly.statistical_signals)),
    }
    return InvestigationSignal(
        code=SignalCode.ANOMALOUS_PATTERN,
        category=SignalCategory.ANOMALY_DETECTION,
        severity=SignalSeverity.MEDIUM,
        title="Statistically atypical pattern",
        explanation=(
            "The anomaly detector marked this claim as statistically atypical "
            "relative to its reference cases. Listed unusual features describe "
            "marginally rare values in the reference data; they are not exact "
            "IsolationForest decision explanations."
        ),
        evidence=evidence,
    )


def _extreme_signal(anomaly, policy):
    if anomaly.reference_percentile is None or anomaly.reference_percentile < policy.extreme_anomaly_percentile:
        return None
    return InvestigationSignal(
        code=SignalCode.EXTREME_ANOMALY_PERCENTILE,
        category=SignalCategory.ANOMALY_DETECTION,
        severity=SignalSeverity.HIGH,
        title="Extreme anomaly percentile",
        explanation=(
            "The anomaly score's reference percentile meets or exceeds the policy's extreme percentile threshold."
        ),
        evidence={
            "anomaly_score": anomaly.anomaly_score,
            "threshold": anomaly.threshold,
            "is_anomalous": anomaly.is_anomalous,
            "reference_percentile": anomaly.reference_percentile,
            "policy_threshold": policy.extreme_anomaly_percentile,
        },
    )
