"""Test money."""

from decimal import Decimal

from persistence.history import HistoricalClaimRepository
from persistence.repositories import ClaimRepository
from tests.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.investigation.history_postgres._support import add_claim


def test_most_recent_date_and_decimal_money_with_currency_isolation(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(
            session,
            reference_claim,
            "eur-older",
            40,
            reference_claim.repair_estimate.model_validate({"amount": "0.10", "currency": "EUR"}),
        )
        add_claim(
            session,
            reference_claim,
            "eur-recent",
            2,
            reference_claim.repair_estimate.model_validate({"amount": "0.20", "currency": "EUR"}),
        )
        add_claim(
            session,
            reference_claim,
            "usd",
            1,
            reference_claim.repair_estimate.model_validate({"amount": "9007199254740993.12345678", "currency": "USD"}),
        )
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == 3
        assert facts.days_since_previous_claim == 1
        assert facts.repair_amount_currency == "EUR"
        assert facts.previous_claims_in_amount_currency == 2
        assert facts.total_previous_repair_amount == Decimal("0.30")
        assert facts.average_previous_repair_amount == Decimal("0.15")
        assert facts.max_previous_repair_amount == Decimal("0.20")


def test_no_compatible_currency_preserves_counts_with_null_amounts(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(
            session,
            reference_claim,
            "usd-only",
            1,
            reference_claim.repair_estimate.model_validate({"amount": "999", "currency": "USD"}),
        )
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == 1
        assert facts.previous_claims_in_amount_currency == 0
        assert facts.total_previous_repair_amount == Decimal("0")
        assert facts.average_previous_repair_amount is facts.max_previous_repair_amount is None
