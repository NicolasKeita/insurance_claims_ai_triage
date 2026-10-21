"""Test anomaly."""

import pytest

from claims.investigation import (
    DEFAULT_INVESTIGATION_POLICY,
    ReviewPriority,
    SignalCode,
)
from claims.investigation.signals import (
    signals_from_anomaly,
)
from tests.domain.investigation.policy._support import make_anomaly


def test_anomaly_is_only_signalled_when_flagged():
    assert signals_from_anomaly(make_anomaly()) == ()

    signals = signals_from_anomaly(
        make_anomaly(
            is_anomalous=True,
            anomaly_score=0.72,
            reference_percentile=0.995,
            statistical_signals=["vehicle_age: very_high"],
        )
    )
    by_code = {s.code: s for s in signals}
    evidence = by_code[SignalCode.ANOMALOUS_PATTERN].evidence
    assert evidence["anomaly_score"] == pytest.approx(0.72)
    assert evidence["threshold"] == pytest.approx(0.62)
    assert evidence["reference_percentile"] == pytest.approx(0.995)
    assert evidence["unusual_features"] == ["vehicle_age: very_high"]
    assert SignalCode.EXTREME_ANOMALY_PERCENTILE in by_code
    assert "fraud" not in by_code[SignalCode.ANOMALOUS_PATTERN].explanation.lower()


def test_anomaly_extreme_percentile_uses_policy_boundary():
    policy = DEFAULT_INVESTIGATION_POLICY
    below = signals_from_anomaly(
        make_anomaly(
            is_anomalous=True,
            anomaly_score=0.72,
            reference_percentile=policy.extreme_anomaly_percentile - 0.001,
        ),
        policy,
    )
    at_boundary = signals_from_anomaly(
        make_anomaly(
            is_anomalous=True,
            anomaly_score=0.72,
            reference_percentile=policy.extreme_anomaly_percentile,
        ),
        policy,
    )
    assert SignalCode.EXTREME_ANOMALY_PERCENTILE not in {s.code for s in below}
    assert SignalCode.EXTREME_ANOMALY_PERCENTILE in {s.code for s in at_boundary}


def test_extreme_percentile_can_be_observed_without_model_flag():
    signals = signals_from_anomaly(make_anomaly(is_anomalous=False, reference_percentile=0.995))
    assert [s.code for s in signals] == [SignalCode.EXTREME_ANOMALY_PERCENTILE]
    policy = DEFAULT_INVESTIGATION_POLICY
    score = policy.score_signals(signals)
    assert policy.priority_for_score(score) != ReviewPriority.NORMAL
    assert policy.review_recommended_for_score(score)
