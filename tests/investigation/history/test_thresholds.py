"""Test thresholds."""

from datetime import date

import pytest

from investigation.history import history_window_start
from investigation.signals import signals_from_history
from tests.investigation.history._support import profile


@pytest.mark.parametrize("days,expected", [(30, date(2026, 8, 21)), (90, date(2026, 6, 22)), (365, date(2025, 9, 20))])
def test_inclusive_window_start(days, expected):
    assert history_window_start(date(2026, 9, 20), days) == expected
    with pytest.raises(ValueError):
        history_window_start(expected, 0)


@pytest.mark.parametrize("count,triggered", [(2, False), (3, True), (4, True)])
def test_frequency_30_day_threshold(count, triggered):
    facts = profile(claims_last_30_days=count, claims_last_365_days=4, previous_anomalous_claim_count=0)
    assert bool(signals_from_history(facts)) is triggered


@pytest.mark.parametrize("count,triggered", [(4, False), (5, True), (6, True)])
def test_frequency_365_day_threshold(count, triggered):
    facts = profile(
        claims_last_30_days=0,
        claims_last_90_days=0,
        claims_last_365_days=count,
        days_since_previous_claim=100,
        previous_anomalous_claim_count=0,
    )
    assert bool(signals_from_history(facts)) is triggered


@pytest.mark.parametrize("count,triggered", [(1, False), (2, True), (3, True)])
def test_anomaly_history_threshold(count, triggered):
    facts = profile(
        claims_last_30_days=0,
        claims_last_90_days=0,
        claims_last_365_days=0,
        days_since_previous_claim=400,
        previous_anomalous_claim_count=count,
    )
    assert bool(signals_from_history(facts)) is triggered
