"""In-memory historical profile boundary built from controlled claims."""

from datetime import timedelta
from decimal import Decimal

from investigation.history import HistoricalClaimProfile
from persistence.repositories import ClaimNotFoundError


def historical_claims(current, fixtures):
    """Make controlled historical inputs; dates are relative to the incident."""
    claims = []
    for index, fixture in enumerate(fixtures):
        payload = current.model_dump(mode="json")
        payload["claim_id"] = fixture.get("claim_id", f"DEMO-HISTORY-{index:03d}")
        payload["incident"]["date"] = (
            current.incident.date - timedelta(days=fixture.get("days_before", 10))
        ).isoformat()
        payload["customer"]["customer_id"] = fixture.get("customer_id", current.customer.customer_id)
        payload["repair_estimate"] = {
            "amount": str(fixture.get("amount", 1000)),
            "currency": fixture.get("currency", current.repair_estimate.currency),
        }
        claims.append((type(current).model_validate(payload), bool(fixture.get("is_anomalous", False))))
    return claims


class FakeHistoricalRepository:
    """Repository boundary implementing incident-time aggregates on test records."""

    def __init__(self, current, claims):
        self.current, self.claims = current, claims

    def _previous(self):
        current = self.current
        return [
            (claim, flag)
            for claim, flag in self.claims
            if claim.customer.customer_id == current.customer.customer_id
            and claim.incident.date < current.incident.date
        ]

    def _amounts(self, previous):
        return [
            claim.repair_estimate.amount
            for claim, _ in previous
            if claim.repair_estimate.currency == self.current.repair_estimate.currency
        ]

    def get_profile(self, claim_id):
        if claim_id != self.current.claim_id:
            raise ClaimNotFoundError(claim_id)
        current = self.current
        previous = self._previous()
        compatible = self._amounts(previous)
        return HistoricalClaimProfile(
            previous_claim_count=len(previous),
            **{
                f"claims_last_{days}_days": sum(
                    (current.incident.date - claim.incident.date).days <= days for claim, _ in previous
                )
                for days in (30, 90, 365)
            },
            days_since_previous_claim=min(
                ((current.incident.date - claim.incident.date).days for claim, _ in previous), default=None
            ),
            repair_amount_currency=current.repair_estimate.currency,
            previous_claims_in_amount_currency=len(compatible),
            total_previous_repair_amount=sum(compatible, Decimal("0")),
            average_previous_repair_amount=sum(compatible) / len(compatible) if compatible else None,
            max_previous_repair_amount=max(compatible) if compatible else None,
            previous_anomalous_claim_count=sum(flag for _, flag in previous),
        )
