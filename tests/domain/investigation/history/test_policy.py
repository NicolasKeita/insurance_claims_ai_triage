"""Test policy."""

import pytest
from pydantic import ValidationError

from claims.investigation.engine import InvestigationContext, assess_investigation
from claims.investigation.models import SignalCategory, SignalCode
from claims.investigation.policy import (
    INVESTIGATION_POLICY_V1,
    INVESTIGATION_POLICY_V2,
    InvestigationPolicy,
)
from claims.investigation.signals import signals_from_history
from tests.domain.investigation.history._support import profile
from tests.domain.investigation.policy._support import (
    make_anomaly,
    make_consistency,
    make_features,
)


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
        SignalCode.DOCUMENT_FIELD_MISMATCH: 10,
        SignalCode.ADDITIONAL_QUOTE_DAMAGE: 18,
        SignalCode.UNMAPPED_QUOTE_ITEM: 8,
        SignalCode.ANOMALOUS_PATTERN: 18,
        SignalCode.EXTREME_ANOMALY_PERCENTILE: 20,
        SignalCode.MISSING_DOCUMENTS: 12,
    }
    assert (
        v1.elevated_priority_min_score,
        v1.high_priority_min_score,
        v1.review_recommendation_min_score,
        v1.max_review_score,
    ) == (20, 50, 20, 100)
    assert (v1.extreme_anomaly_percentile, v1.minimum_document_completeness_ratio) == (0.99, 0.80)
    assert signals_from_history(profile(), v1) == ()
    inputs = dict(consistency=make_consistency(), anomaly=make_anomaly(), features=make_features())
    context = InvestigationContext(history=profile())
    assert assess_investigation(**inputs, policy=v1, context=context).review_score == 0
    first = assess_investigation(**inputs, context=context)
    assert first == assess_investigation(**inputs, context=context)
    assert first.policy_version == "investigation_policy_v2"
    assert first.review_score == 33
    assert INVESTIGATION_POLICY_V2.score_signals(first.signals * 100) == 100
    assert assess_investigation(**inputs).review_score == 0


def test_thresholds_are_configurable_and_v1_cannot_enable_history():
    data = INVESTIGATION_POLICY_V2.model_dump()
    data.update(
        version="custom_demo",
        recent_claims_30_days_threshold=4,
        recent_claims_365_days_threshold=6,
        previous_anomalous_claims_threshold=3,
    )
    assert signals_from_history(profile(), InvestigationPolicy(**data)) == ()
    data["version"] = "investigation_policy_v1"
    with pytest.raises(ValidationError):
        InvestigationPolicy(**data)


def test_cumulative_amount_alone_does_not_create_signal():
    facts = profile(
        claims_last_30_days=0,
        claims_last_90_days=0,
        claims_last_365_days=0,
        days_since_previous_claim=400,
        previous_anomalous_claim_count=0,
        repair_amount_currency="EUR",
        previous_claims_in_amount_currency=6,
        total_previous_repair_amount="6000000",
        average_previous_repair_amount="1000000",
        max_previous_repair_amount="1000000",
    )
    assert signals_from_history(facts) == ()
