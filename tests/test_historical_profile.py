"""Pure domain and policy rules; no database or model server needed."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from investigation.engine import assess_investigation
from investigation.history import HistoricalClaimProfile, history_window_start
from investigation.models import SignalCategory, SignalCode
from investigation.policy import INVESTIGATION_POLICY_V1, INVESTIGATION_POLICY_V2, InvestigationPolicy
from investigation.signals import signals_from_history
from test_investigation import make_anomaly, make_consistency, make_features


def profile(**changes):
    return HistoricalClaimProfile(**{
        "previous_claim_count": 6, "claims_last_30_days": 3,
        "claims_last_90_days": 4, "claims_last_365_days": 5,
        "days_since_previous_claim": 2, "previous_anomalous_claim_count": 2,
        **changes,
    })


def test_empty_profile_and_decimal_semantics():
    empty = HistoricalClaimProfile()
    assert empty.previous_claim_count == empty.previous_anomalous_claim_count == 0
    assert empty.total_previous_repair_amount == Decimal("0")
    assert isinstance(empty.total_previous_repair_amount, Decimal)
    assert empty.average_previous_repair_amount is empty.max_previous_repair_amount is None
    assert empty.days_since_previous_claim is None
    assert signals_from_history(empty) == ()
    assert HistoricalClaimProfile.model_validate_json(empty.model_dump_json()) == empty


@pytest.mark.parametrize("changes", [
    {"previous_claim_count": -1}, {"claims_last_30_days": 7},
    {"previous_anomalous_claim_count": 7}, {"days_since_previous_claim": 0},
    {"days_since_previous_claim": None}, {"days_since_previous_claim": 31},
    {"previous_claims_in_amount_currency": 7}, {"total_previous_repair_amount": "1"},
    {"total_previous_repair_amount": "NaN"}, {"total_previous_repair_amount": "Infinity"},
    {"repair_amount_currency": "eur"}, {"previous_claims_in_amount_currency": 1},
])
def test_incoherent_profile_rejected(changes):
    with pytest.raises(ValidationError):
        profile(**changes)


def test_money_profile_retains_decimal_precision():
    facts = profile(
        repair_amount_currency="EUR", previous_claims_in_amount_currency=2,
        total_previous_repair_amount="0.30", average_previous_repair_amount="0.15",
        max_previous_repair_amount="0.20",
    )
    assert facts.total_previous_repair_amount == Decimal("0.30")
    assert HistoricalClaimProfile.model_validate_json(facts.model_dump_json()) == facts


@pytest.mark.parametrize("days,expected", [(30, date(2026, 8, 21)), (90, date(2026, 6, 22)), (365, date(2025, 9, 20))])
def test_inclusive_window_start(days, expected):
    assert history_window_start(date(2026, 9, 20), days) == expected
    with pytest.raises(ValueError):
        history_window_start(expected, 0)


@pytest.mark.parametrize("count,triggered", [(2, False), (3, True), (4, True)])
def test_frequency_30_day_threshold(count, triggered):
    facts = profile(claims_last_30_days=count, claims_last_365_days=4,
                    previous_anomalous_claim_count=0)
    assert bool(signals_from_history(facts)) is triggered


@pytest.mark.parametrize("count,triggered", [(4, False), (5, True), (6, True)])
def test_frequency_365_day_threshold(count, triggered):
    facts = profile(claims_last_30_days=0, claims_last_90_days=0,
                    claims_last_365_days=count, days_since_previous_claim=100,
                    previous_anomalous_claim_count=0)
    assert bool(signals_from_history(facts)) is triggered


@pytest.mark.parametrize("count,triggered", [(1, False), (2, True), (3, True)])
def test_anomaly_history_threshold(count, triggered):
    facts = profile(claims_last_30_days=0, claims_last_90_days=0,
                    claims_last_365_days=0, days_since_previous_claim=400,
                    previous_anomalous_claim_count=count)
    assert bool(signals_from_history(facts)) is triggered


def test_explanations_evidence_and_single_frequency_weight():
    signals = signals_from_history(profile())
    assert [s.code for s in signals] == [SignalCode.RECENT_CLAIM_FREQUENCY, SignalCode.PREVIOUS_ANOMALOUS_CLAIMS]
    assert all(s.category == SignalCategory.HISTORICAL_CONTEXT for s in signals)
    assert signals[0].evidence["claims_last_30_days"] == 3
    assert signals[0].evidence["claims_last_365_days"] == 5
    assert signals[1].evidence["previous_anomalous_claim_count"] == 2
    assert all(s.evidence["previous_claim_count"] == 6 for s in signals)
    assert all("fraud" not in (s.title + s.explanation).lower() for s in signals)
    assert INVESTIGATION_POLICY_V2.score_signals(signals) == 33


def test_v1_original_rules_and_v2_determinism():
    v1 = INVESTIGATION_POLICY_V1
    assert v1 == InvestigationPolicy()
    assert v1.version == "investigation_policy_v1"
    assert dict(v1.signal_weights) == {
        SignalCode.DOCUMENT_FIELD_MISMATCH: 10, SignalCode.ADDITIONAL_QUOTE_DAMAGE: 18,
        SignalCode.UNMAPPED_QUOTE_ITEM: 8, SignalCode.ANOMALOUS_PATTERN: 18,
        SignalCode.EXTREME_ANOMALY_PERCENTILE: 20, SignalCode.MISSING_DOCUMENTS: 12,
    }
    assert (v1.elevated_priority_min_score, v1.high_priority_min_score,
            v1.review_recommendation_min_score, v1.max_review_score) == (20, 50, 20, 100)
    assert (v1.extreme_anomaly_percentile, v1.minimum_document_completeness_ratio) == (0.99, 0.80)
    assert signals_from_history(profile(), v1) == ()
    inputs = dict(consistency=make_consistency(), anomaly=make_anomaly(), features=make_features(), history=profile())
    assert assess_investigation(**inputs, policy=v1).review_score == 0
    first = assess_investigation(**inputs)
    assert first == assess_investigation(**inputs)
    assert first.policy_version == "investigation_policy_v2"
    assert first.review_score == 33
    assert INVESTIGATION_POLICY_V2.score_signals(first.signals * 100) == 100
    assert assess_investigation(**{**inputs, "history": None}).review_score == 0


def test_thresholds_are_configurable_and_v1_cannot_enable_history():
    data = INVESTIGATION_POLICY_V2.model_dump()
    data.update(version="custom_demo", recent_claims_30_days_threshold=4,
                recent_claims_365_days_threshold=6, previous_anomalous_claims_threshold=3)
    assert signals_from_history(profile(), InvestigationPolicy(**data)) == ()
    data["version"] = "investigation_policy_v1"
    with pytest.raises(ValidationError):
        InvestigationPolicy(**data)


def test_cumulative_amount_alone_does_not_create_signal():
    facts = profile(claims_last_30_days=0, claims_last_90_days=0, claims_last_365_days=0,
                    days_since_previous_claim=400, previous_anomalous_claim_count=0,
                    repair_amount_currency="EUR", previous_claims_in_amount_currency=6,
                    total_previous_repair_amount="6000000", average_previous_repair_amount="1000000",
                    max_previous_repair_amount="1000000")
    assert signals_from_history(facts) == ()
