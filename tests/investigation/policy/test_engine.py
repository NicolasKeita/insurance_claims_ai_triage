"""Test engine."""

import inspect

from claims.enums import DamageType
from investigation import (
    DEFAULT_INVESTIGATION_POLICY,
    ReviewPriority,
    assess_investigation,
)
from tests.investigation.policy._support import (
    make_anomaly,
    make_consistency,
    make_features,
)


def test_engine_is_reproducible_bounded_and_keeps_signal_order():
    policy = DEFAULT_INVESTIGATION_POLICY
    consistency = make_consistency(
        incident_date_match=False,
        vehicle_match=False,
        additional_quote_damage=(DamageType.HOOD,),
        unmapped_quote_items=("Paint labor",),
    )
    anomaly = make_anomaly(
        is_anomalous=True,
        anomaly_score=0.72,
        reference_percentile=0.995,
    )
    features = make_features(
        document_count=4,
        missing_document_count=3,
        document_completeness_ratio=0.25,
    )
    first = assess_investigation(
        consistency=consistency,
        anomaly=anomaly,
        features=features,
        policy=policy,
    )
    second = assess_investigation(
        consistency=consistency,
        anomaly=anomaly,
        features=features,
        policy=policy,
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert 0 <= first.review_score <= 100
    assert first.policy_version == policy.version
    assert first.review_priority == ReviewPriority.HIGH
    assert [s.code for s in first.signals] == [s.code for s in second.signals]
    assert policy.score_signals(first.signals * 100) == policy.max_review_score
    assert "fraud_probability" not in first.model_dump_json()
    assert "fraud_detected" not in first.model_dump_json()


def test_no_signals_normal_and_multiple_signals_raise_priority():
    baseline = assess_investigation(
        consistency=make_consistency(),
        anomaly=make_anomaly(),
        features=make_features(),
    )
    several = assess_investigation(
        consistency=make_consistency(
            claim_id_match=False,
            incident_date_match=False,
            vehicle_match=False,
            injuries_match=False,
        ),
        anomaly=make_anomaly(),
        features=make_features(),
    )
    assert baseline.signals == ()
    assert baseline.review_score == 0
    assert baseline.review_priority == ReviewPriority.NORMAL
    assert baseline.review_recommended is False
    assert several.review_score > baseline.review_score
    assert several.review_priority != ReviewPriority.NORMAL


def test_high_repair_amount_or_declared_injuries_alone_do_not_create_signals():
    assessment = assess_investigation(
        consistency=make_consistency(),
        anomaly=make_anomaly(),
        features=make_features(repair_amount=1_000_000, injuries_declared=True),
    )
    assert assessment.signals == ()
    assert assessment.review_score == 0


def test_engine_has_no_triage_workflow_input():
    assert set(inspect.signature(assess_investigation).parameters) == {
        "consistency",
        "anomaly",
        "features",
        "policy",
        "context",
    }
