"""Test temporal."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select

from persistence.history import HistoricalClaimRepository
from persistence.models import (
    ClaimRow,
)
from persistence.repositories import ClaimRepository
from tests.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.investigation.history_postgres._support import add_claim


def test_temporal_customer_same_day_and_created_at_independence(factory, reference_claim):
    """
    Insertion time deliberately contradicts incident chronology.
    """
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(session, reference_claim, "previous", 2)
        add_claim(session, reference_claim, "later", -1)
        add_claim(session, reference_claim, "same-day", 0)
        add_claim(
            session,
            reference_claim.model_copy(
                update={"customer": reference_claim.customer.model_validate({"customer_id": "OTHER"})}
            ),
            "other-customer",
            1,
        )
        current = session.scalar(select(ClaimRow).where(ClaimRow.claim_id == reference_claim.claim_id))
        current.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == facts.claims_last_30_days == 1
        assert facts.days_since_previous_claim == 2
        assert facts.total_previous_repair_amount == Decimal("100")


@pytest.mark.parametrize(
    "offset,counts",
    [
        (0, (0, 0, 0)),
        (1, (1, 1, 1)),
        (29, (1, 1, 1)),
        (30, (1, 1, 1)),
        (31, (0, 1, 1)),
        (89, (0, 1, 1)),
        (90, (0, 1, 1)),
        (91, (0, 0, 1)),
        (364, (0, 0, 1)),
        (365, (0, 0, 1)),
        (366, (0, 0, 0)),
    ],
)
def test_precise_window_boundaries(factory, reference_claim, offset, counts):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(session, reference_claim, f"boundary-{offset}", offset)
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert (facts.claims_last_30_days, facts.claims_last_90_days, facts.claims_last_365_days) == counts
        assert facts.days_since_previous_claim == (offset or None)
