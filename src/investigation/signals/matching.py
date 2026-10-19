"""Document-field match checks that become mismatch signals."""

from claims.consistency import ClaimConsistencyReport
from investigation.models import InvestigationSignal, SignalCategory, SignalCode, SignalSeverity

# The report currently exposes matches, rather than the underlying compared values.
# An observed False supports a mismatch, but not a claim about which value is correct.
MATCH_CHECKS = (
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


def mismatch_signals(consistency: ClaimConsistencyReport) -> list[InvestigationSignal]:
    """Preserve which document checks failed for one consistency report."""
    signals: list[InvestigationSignal] = []
    for attribute, field, title, sources in MATCH_CHECKS:
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
                explanation=(f"The {field.replace('_', ' ')} values do not match across {readable_sources}."),
                evidence={
                    "field": field,
                    "matched": False,
                    "compared_sources": list(sources),
                },
            )
        )
    return signals
