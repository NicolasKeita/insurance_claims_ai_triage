"""Independent fact-to-signal translators. Scoring belongs to the policy."""

from claims.consistency import ClaimConsistencyReport
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures

from investigation.models import (
    InvestigationSignal,
    SignalCategory,
    SignalCode,
    SignalSeverity,
)
from investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy
from investigation.history import HistoricalClaimProfile


def signals_from_history(
    history: HistoricalClaimProfile,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
) -> tuple[InvestigationSignal, ...]:
    """Translate historical facts once per rule; never consume prior review scores."""
    signals = []
    windows = (
        (30, history.claims_last_30_days, policy.recent_claims_30_days_threshold),
        (365, history.claims_last_365_days, policy.recent_claims_365_days_threshold),
    )
    triggered = [(days, count) for days, count, threshold in windows
                 if threshold is not None and count >= threshold]
    if triggered:
        signals.append(InvestigationSignal(
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
        ))
    if (policy.previous_anomalous_claims_threshold is not None
            and history.previous_anomalous_claim_count >= policy.previous_anomalous_claims_threshold):
        signals.append(InvestigationSignal(
            code=SignalCode.PREVIOUS_ANOMALOUS_CLAIMS,
            category=SignalCategory.HISTORICAL_CONTEXT,
            severity=SignalSeverity.MEDIUM,
            title="Previous anomaly assessments",
            explanation=(f"{history.previous_anomalous_claim_count} previous claims are currently "
                         "flagged as anomalous by their latest stored anomaly assessments."),
            evidence={
                "previous_anomalous_claim_count": history.previous_anomalous_claim_count,
                "previous_claim_count": history.previous_claim_count,
                "policy_threshold": policy.previous_anomalous_claims_threshold,
            },
        ))
    return tuple(signals)


# The report currently exposes matches, rather than the underlying compared values.
# An observed False supports a mismatch, but not a claim about which value is correct.
_MATCH_CHECKS = (
    (
        "claim_id_match",
        "claim_id",
        "Claim identifiers",
        ("claim_form", "accident_report", "repair_quote"),
    ),
    (
        "incident_date_match",
        "incident_date",
        "Incident dates",
        ("claim_form", "accident_report"),
    ),
    (
        "location_match",
        "location",
        "Incident locations",
        ("claim_form", "accident_report"),
    ),
    (
        "vehicle_match",
        "vehicle",
        "Vehicle details",
        ("claim_form", "accident_report"),
    ),
    (
        "collision_type_match",
        "collision_type",
        "Collision types",
        ("claim_form", "accident_report"),
    ),
    (
        "injuries_match",
        "injuries_declared",
        "Injury declarations",
        ("claim_form", "accident_report"),
    ),
    (
        "declared_damage_match",
        "declared_damage",
        "Declared damage",
        ("claim_form", "accident_report"),
    ),
)


def signals_from_consistency(
    consistency: ClaimConsistencyReport,
) -> tuple[InvestigationSignal, ...]:
    """Preserve which document checks failed and which quote items were observed."""

    signals: list[InvestigationSignal] = []
    for attribute, field, title, sources in _MATCH_CHECKS:
        if getattr(consistency, attribute):
            continue
        source_names = [source.replace("_", " ") for source in sources]
        readable_sources = (
            " and ".join(source_names)
            if len(source_names) == 2
            else ", ".join(source_names[:-1]) + ", and " + source_names[-1]
        )
        signals.append(
            InvestigationSignal(
                code=SignalCode.DOCUMENT_FIELD_MISMATCH,
                category=SignalCategory.DOCUMENT_CONSISTENCY,
                severity=SignalSeverity.MEDIUM,
                title=f"{title} do not match",
                explanation=(
                    f"The {field.replace('_', ' ')} values do not match across "
                    f"{readable_sources}."
                ),
                evidence={
                    "field": field,
                    "matched": False,
                    "compared_sources": list(sources),
                },
            )
        )

    damage_types = sorted(damage.value for damage in consistency.additional_quote_damage)
    if damage_types:
        count = len(damage_types)
        signals.append(
            InvestigationSignal(
                code=SignalCode.ADDITIONAL_QUOTE_DAMAGE,
                category=SignalCategory.DOCUMENT_CONSISTENCY,
                severity=SignalSeverity.MEDIUM,
                title="Additional damage in repair quote",
                explanation=(
                    f"{count} damage type{'s' if count != 1 else ''} "
                    f"{'appear' if count != 1 else 'appears'} in the "
                    "repair quote but not in the claim form."
                ),
                evidence={"damage_types": damage_types, "count": count},
            )
        )

    descriptions = sorted(
        consistency.unmapped_quote_items,
        key=lambda value: (value.casefold(), value),
    )
    if descriptions:
        count = len(descriptions)
        signals.append(
            InvestigationSignal(
                code=SignalCode.UNMAPPED_QUOTE_ITEM,
                category=SignalCategory.DOCUMENT_CONSISTENCY,
                severity=SignalSeverity.LOW,
                title="Unmapped repair quote items",
                explanation=(
                    f"{count} repair quote item{'s' if count != 1 else ''} could not be "
                    "mapped to the current damage categories."
                ),
                evidence={"descriptions": descriptions, "count": count},
            )
        )
    return tuple(signals)


def signals_from_anomaly(
    anomaly: AnomalyAssessment,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
) -> tuple[InvestigationSignal, ...]:
    """Report a model flag and independent rarity descriptors, without attribution claims."""

    unusual_features = sorted(set(anomaly.statistical_signals))
    evidence = {
        "anomaly_score": anomaly.anomaly_score,
        "threshold": anomaly.threshold,
        "reference_percentile": anomaly.reference_percentile,
        "unusual_features": unusual_features,
    }
    signals: list[InvestigationSignal] = []
    if anomaly.is_anomalous:
        signals.append(
            InvestigationSignal(
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
        )

    if (
        anomaly.reference_percentile is not None
        and anomaly.reference_percentile >= policy.extreme_anomaly_percentile
    ):
        signals.append(
            InvestigationSignal(
                code=SignalCode.EXTREME_ANOMALY_PERCENTILE,
                category=SignalCategory.ANOMALY_DETECTION,
                severity=SignalSeverity.HIGH,
                title="Extreme anomaly percentile",
                explanation=(
                    "The anomaly score's reference percentile meets or exceeds "
                    "the policy's extreme percentile threshold."
                ),
                evidence={
                    "anomaly_score": anomaly.anomaly_score,
                    "threshold": anomaly.threshold,
                    "is_anomalous": anomaly.is_anomalous,
                    "reference_percentile": anomaly.reference_percentile,
                    "policy_threshold": policy.extreme_anomaly_percentile,
                },
            )
        )
    return tuple(signals)


def signals_from_completeness(
    features: TriageFeatures,
    policy: InvestigationPolicy = DEFAULT_INVESTIGATION_POLICY,
) -> tuple[InvestigationSignal, ...]:
    """Describe unavailable dossier entries, without claiming they are mandatory."""

    count = features.document_count
    missing = features.missing_document_count
    ratio = features.document_completeness_ratio
    if count == 0 or missing == 0 or ratio >= policy.minimum_document_completeness_ratio:
        return ()

    return (
        InvestigationSignal(
            code=SignalCode.MISSING_DOCUMENTS,
            category=SignalCategory.DATA_COMPLETENESS,
            severity=SignalSeverity.LOW,
            title="Dossier entries unavailable",
            explanation=(
                f"{missing} of {count} dossier document entries "
                f"{'are' if missing != 1 else 'is'} currently marked unavailable."
            ),
            evidence={
                "document_count": count,
                "missing_document_count": missing,
                "document_completeness_ratio": ratio,
                "minimum_document_completeness_ratio": (
                    policy.minimum_document_completeness_ratio
                ),
            },
        ),
    )
