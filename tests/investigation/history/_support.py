"""Support."""

from investigation.history import HistoricalClaimProfile


def profile(**changes):
    return HistoricalClaimProfile(
        **{
            "previous_claim_count": 6,
            "claims_last_30_days": 3,
            "claims_last_90_days": 4,
            "claims_last_365_days": 5,
            "days_since_previous_claim": 2,
            "previous_anomalous_claim_count": 2,
            **changes,
        }
    )
