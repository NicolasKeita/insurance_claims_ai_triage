"""Service-free checks for deterministic investigation signals and policy."""

import inspect
import json

import pytest

from claims.consistency import ClaimConsistencyReport
from claims.enums import DamageType
from investigation import (
    DEFAULT_INVESTIGATION_POLICY,
    InvestigationPolicy,
    ReviewPriority,
    SignalCode,
    assess_investigation,
)
from investigation.signals import (
    signals_from_anomaly,
    signals_from_completeness,
    signals_from_consistency,
)
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures


def make_consistency(**changes) -> ClaimConsistencyReport:
    values = dict(
        claim_id_match=True,
        incident_date_match=True,
        location_match=True,
        vehicle_match=True,
        collision_type_match=True,
        injuries_match=True,
        declared_damage_match=True,
        additional_quote_damage=(),
        unmapped_quote_items=(),
    )
    values.update(changes)
    return ClaimConsistencyReport(**values)


def make_anomaly(**changes) -> AnomalyAssessment:
    values = dict(
        is_anomalous=False,
        anomaly_score=0.42,
        threshold=0.62,
        reference_percentile=0.75,
        statistical_signals=[],
    )
    values.update(changes)
    return AnomalyAssessment(**values)


def make_features(**changes) -> TriageFeatures:
    values = dict(
        claim_type="AUTO_COLLISION",
        collision_type="FRONT_COLLISION",
        vehicle_age=5,
        repair_amount=3160,
        declared_damage_count=2,
        additional_damage_count=0,
        document_count=4,
        missing_document_count=0,
        document_completeness_ratio=1.0,
        injuries_declared=False,
        consistency_issue_count=0,
    )
    values.update(changes)
    return TriageFeatures(**values)


def test_perfect_consistency_has_no_signals():
    assert signals_from_consistency(make_consistency()) == ()


@pytest.mark.parametrize(
    "check, field",
    [
        ("claim_id_match", "claim_id"),
        ("incident_date_match", "incident_date"),
        ("location_match", "location"),
        ("vehicle_match", "vehicle"),
        ("collision_type_match", "collision_type"),
        ("injuries_match", "injuries_declared"),
        ("declared_damage_match", "declared_damage"),
    ],
)
def test_each_consistency_check_has_an_explicit_field_signal(check, field):
    signals = signals_from_consistency(make_consistency(**{check: False}))
    assert len(signals) == 1
    assert signals[0].code == SignalCode.DOCUMENT_FIELD_MISMATCH
    assert signals[0].evidence["field"] == field


def test_each_document_mismatch_keeps_its_field_in_evidence():
    signals = signals_from_consistency(
        make_consistency(incident_date_match=False, vehicle_match=False)
    )
    mismatches = [s for s in signals if s.code == SignalCode.DOCUMENT_FIELD_MISMATCH]
    assert {s.evidence["field"] for s in mismatches} == {
        "incident_date",
        "vehicle",
    }
    assert all(s.evidence["matched"] is False for s in mismatches)
    assert all(s.explanation for s in mismatches)


def test_quote_damage_and_unmapped_descriptions_are_json_evidence():
    signals = signals_from_consistency(
        make_consistency(
            additional_quote_damage=(DamageType.HOOD,),
            unmapped_quote_items=("Paint labor", "Diagnostic scan"),
        )
    )
    by_code = {s.code: s for s in signals}
    assert by_code[SignalCode.ADDITIONAL_QUOTE_DAMAGE].evidence["damage_types"] == [
        "HOOD"
    ]
    assert set(by_code[SignalCode.UNMAPPED_QUOTE_ITEM].evidence["descriptions"]) == {
        "Paint labor",
        "Diagnostic scan",
    }
    json.dumps([s.model_dump(mode="json") for s in signals])


def test_repeated_unmapped_quote_items_keep_each_observed_occurrence():
    signals = signals_from_consistency(
        make_consistency(unmapped_quote_items=("Paint labor", "Paint labor"))
    )
    assert len(signals) == 1
    assert signals[0].code == SignalCode.UNMAPPED_QUOTE_ITEM
    assert signals[0].evidence["descriptions"] == ["Paint labor", "Paint labor"]
    assert signals[0].evidence["count"] == 2


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
    signals = signals_from_anomaly(
        make_anomaly(is_anomalous=False, reference_percentile=0.995)
    )
    assert [s.code for s in signals] == [SignalCode.EXTREME_ANOMALY_PERCENTILE]
    policy = DEFAULT_INVESTIGATION_POLICY
    score = policy.score_signals(signals)
    assert policy.priority_for_score(score) != ReviewPriority.NORMAL
    assert policy.review_recommended_for_score(score)


def test_missing_documents_only_below_policy_completeness_threshold():
    policy = DEFAULT_INVESTIGATION_POLICY
    complete = signals_from_completeness(make_features(), policy)
    at_boundary = signals_from_completeness(
        make_features(
            document_count=5,
            missing_document_count=1,
            document_completeness_ratio=policy.minimum_document_completeness_ratio,
        ),
        policy,
    )
    below = signals_from_completeness(
        make_features(
            document_count=4,
            missing_document_count=1,
            document_completeness_ratio=0.75,
        ),
        policy,
    )
    assert complete == ()
    assert at_boundary == ()
    assert [s.code for s in below] == [SignalCode.MISSING_DOCUMENTS]
    assert below[0].evidence["missing_document_count"] == 1
    assert below[0].evidence["document_count"] == 4
    assert "required" not in below[0].explanation.lower()


def test_policy_priority_boundaries_and_recommendation():
    policy = DEFAULT_INVESTIGATION_POLICY
    assert policy.priority_for_score(0) == ReviewPriority.NORMAL
    assert policy.priority_for_score(policy.elevated_priority_min_score - 1) == ReviewPriority.NORMAL
    assert policy.priority_for_score(policy.elevated_priority_min_score) == ReviewPriority.ELEVATED
    assert policy.priority_for_score(policy.high_priority_min_score - 1) == ReviewPriority.ELEVATED
    assert policy.priority_for_score(policy.high_priority_min_score) == ReviewPriority.HIGH
    assert not policy.review_recommended_for_score(policy.review_recommendation_min_score - 1)
    assert policy.review_recommended_for_score(policy.review_recommendation_min_score)


def test_versioned_policy_weights_are_immutable_and_json_serializable():
    default = DEFAULT_INVESTIGATION_POLICY
    weights = dict(default.signal_weights)
    weights[SignalCode.UNMAPPED_QUOTE_ITEM] = 9
    custom = InvestigationPolicy(
        **{**default.model_dump(), "version": "custom_test_policy", "signal_weights": weights}
    )
    with pytest.raises(TypeError):
        default.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] = 9
    with pytest.raises(TypeError):
        custom.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] = 10
    assert default.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] == 8
    assert custom.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] == 9
    json.dumps(custom.model_dump(mode="json"))


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
        "history",
    }
