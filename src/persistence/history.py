"""Historical aggregates from PostgreSQL, with two bounded round trips."""

from decimal import Decimal

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from claims.investigation.history import HistoricalClaimProfile, history_window_start
from persistence.models import AnomalyAssessmentRow, ClaimRow
from persistence.repositories import ClaimNotFoundError


def previous_claim_predicate(current):
    """Single business-time filter; insertion timestamps never define history."""
    return and_(
        ClaimRow.customer_pk == current.customer_pk,
        ClaimRow.incident_date < current.incident_date,
    )


class HistoricalClaimRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_profile(self, claim_id: str) -> HistoricalClaimProfile:
        """
        Rank only assessments of eligible claims. One row per claim is joined, so append-only
        assessment histories cannot multiply claim aggregates.
        """
        current = self._current_claim(claim_id)
        previous = self._previous_claims(current)
        latest = self._latest_anomalies(previous)
        values = dict(self.session.execute(self._aggregate_query(previous, latest, current)).one()._mapping)
        return self._profile(values, current)

    def _current_claim(self, claim_id: str):
        current = self.session.execute(
            select(
                ClaimRow.customer_pk,
                ClaimRow.incident_date,
                ClaimRow.repair_estimate_currency,
            ).where(ClaimRow.claim_id == claim_id)
        ).one_or_none()
        if current is None:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return current

    @staticmethod
    def _previous_claims(current):
        return (
            select(
                ClaimRow.id,
                ClaimRow.incident_date,
                ClaimRow.repair_estimate_amount,
                ClaimRow.repair_estimate_currency,
            )
            .where(previous_claim_predicate(current))
            .cte("previous_claims")
        )

    @staticmethod
    def _latest_anomalies(previous):
        return (
            select(
                AnomalyAssessmentRow.claim_pk,
                AnomalyAssessmentRow.is_anomalous,
                func.row_number()
                .over(
                    partition_by=AnomalyAssessmentRow.claim_pk,
                    order_by=(AnomalyAssessmentRow.created_at.desc(), AnomalyAssessmentRow.insertion_order.desc()),
                )
                .label("rank"),
            )
            .join(previous, previous.c.id == AnomalyAssessmentRow.claim_pk)
            .subquery()
        )

    @staticmethod
    def _aggregate_query(previous, latest, current):
        same_currency = previous.c.repair_estimate_currency == current.repair_estimate_currency
        return select(
            func.count(previous.c.id).label("previous_claim_count"),
            *(
                func.count(previous.c.id)
                .filter(previous.c.incident_date >= history_window_start(current.incident_date, days))
                .label(f"claims_last_{days}_days")
                for days in (30, 90, 365)
            ),
            func.max(previous.c.incident_date).label("latest_incident"),
            func.count(previous.c.id).filter(same_currency).label("previous_claims_in_amount_currency"),
            func.sum(previous.c.repair_estimate_amount).filter(same_currency).label("total_previous_repair_amount"),
            func.avg(previous.c.repair_estimate_amount).filter(same_currency).label("average_previous_repair_amount"),
            func.max(previous.c.repair_estimate_amount).filter(same_currency).label("max_previous_repair_amount"),
            func.count(previous.c.id).filter(latest.c.is_anomalous.is_(True)).label("previous_anomalous_claim_count"),
        ).select_from(
            previous.outerjoin(
                latest,
                and_(
                    latest.c.claim_pk == previous.c.id,
                    latest.c.rank == 1,
                ),
            )
        )

    @staticmethod
    def _profile(values, current):
        latest_date = values.pop("latest_incident")
        values["days_since_previous_claim"] = (current.incident_date - latest_date).days if latest_date else None
        values["total_previous_repair_amount"] = values["total_previous_repair_amount"] or Decimal("0")
        values["repair_amount_currency"] = current.repair_estimate_currency
        return HistoricalClaimProfile(**values)
