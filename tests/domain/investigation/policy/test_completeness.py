"""Test completeness."""

import json

import pytest

from claims.investigation import (
    DEFAULT_INVESTIGATION_POLICY,
    InvestigationPolicy,
    ReviewPriority,
    SignalCode,
)
from claims.investigation.signals import (
    signals_from_completeness,
)
from tests.domain.investigation.policy._support import make_features


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
    custom = InvestigationPolicy(**{**default.model_dump(), "version": "custom_test_policy", "signal_weights": weights})
    with pytest.raises(TypeError):
        default.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] = 9
    with pytest.raises(TypeError):
        custom.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] = 10
    assert default.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] == 8
    assert custom.signal_weights[SignalCode.UNMAPPED_QUOTE_ITEM] == 9
    json.dumps(custom.model_dump(mode="json"))
