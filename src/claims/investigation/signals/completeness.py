"""Describe unavailable dossier entries as a low-severity signal."""

from claims.investigation.models import InvestigationSignal, SignalCategory, SignalCode, SignalSeverity
from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY, InvestigationPolicy
from ml.features import TriageFeatures


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
                "minimum_document_completeness_ratio": (policy.minimum_document_completeness_ratio),
            },
        ),
    )
