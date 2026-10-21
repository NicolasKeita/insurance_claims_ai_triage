"""Translate consistency facts into document and quote signals."""

from claims.consistency import ClaimConsistencyReport
from claims.investigation.models import InvestigationSignal, SignalCategory, SignalCode, SignalSeverity
from claims.investigation.signals.matching import mismatch_signals


def signals_from_consistency(
    consistency: ClaimConsistencyReport,
) -> tuple[InvestigationSignal, ...]:
    """Preserve which document checks failed and which quote items were observed."""

    signals = mismatch_signals(consistency)
    signals.extend(_additional_damage_signals(consistency))
    signals.extend(_unmapped_item_signals(consistency))
    return tuple(signals)


def _additional_damage_signals(consistency):
    damage_types = sorted(damage.value for damage in consistency.additional_quote_damage)
    if not damage_types:
        return ()
    count = len(damage_types)
    return (
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
        ),
    )


def _unmapped_item_signals(consistency):
    descriptions = sorted(
        consistency.unmapped_quote_items,
        key=lambda value: (value.casefold(), value),
    )
    if not descriptions:
        return ()
    count = len(descriptions)
    return (
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
        ),
    )
