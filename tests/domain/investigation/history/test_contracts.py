"""Test contracts."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from claims.investigation.history import HistoricalClaimProfile
from claims.investigation.signals import signals_from_history
from tests.domain.investigation.history._support import profile


def test_empty_profile_and_decimal_semantics():
    empty = HistoricalClaimProfile()
    assert empty.previous_claim_count == empty.previous_anomalous_claim_count == 0
    assert empty.total_previous_repair_amount == Decimal("0")
    assert isinstance(empty.total_previous_repair_amount, Decimal)
    assert empty.average_previous_repair_amount is empty.max_previous_repair_amount is None
    assert empty.days_since_previous_claim is None
    assert signals_from_history(empty) == ()
    assert HistoricalClaimProfile.model_validate_json(empty.model_dump_json()) == empty


@pytest.mark.parametrize(
    "changes",
    [
        {"previous_claim_count": -1},
        {"claims_last_30_days": 7},
        {"previous_anomalous_claim_count": 7},
        {"days_since_previous_claim": 0},
        {"days_since_previous_claim": None},
        {"days_since_previous_claim": 31},
        {"previous_claims_in_amount_currency": 7},
        {"total_previous_repair_amount": "1"},
        {"total_previous_repair_amount": "NaN"},
        {"total_previous_repair_amount": "Infinity"},
        {"repair_amount_currency": "eur"},
        {"previous_claims_in_amount_currency": 1},
    ],
)
def test_incoherent_profile_rejected(changes):
    with pytest.raises(ValidationError):
        profile(**changes)


def test_money_profile_retains_decimal_precision():
    facts = profile(
        repair_amount_currency="EUR",
        previous_claims_in_amount_currency=2,
        total_previous_repair_amount="0.30",
        average_previous_repair_amount="0.15",
        max_previous_repair_amount="0.20",
    )
    assert facts.total_previous_repair_amount == Decimal("0.30")
    assert HistoricalClaimProfile.model_validate_json(facts.model_dump_json()) == facts
