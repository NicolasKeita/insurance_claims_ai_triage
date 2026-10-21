"""Test recommendations."""

import pytest

from agent.evidence import EvidenceValidationError, validate_recommendation
from agent.models import (
    EvidenceItem,
    RecommendedNextAction,
)
from tests.agent.tools._evidence import recommendation
from tests.agent.tools._repositories import configured as configured


@pytest.mark.parametrize(
    "action", [action for action in RecommendedNextAction if action != RecommendedNextAction.NO_RECOMMENDATION]
)
@pytest.mark.parametrize("top_level_citation", [False, True])
def test_non_abstaining_action_requires_structured_supporting_rationale(action, top_level_citation):
    item = EvidenceItem(evidence_id="claim:demo", source_type="CLAIM_DATA", source_reference="demo", summary="Fact")
    output = recommendation(
        recommended_next_action=action, evidence_ids=[item.evidence_id] if top_level_citation else [], rationale=[]
    )
    with pytest.raises(EvidenceValidationError, match="cited rationale"):
        validate_recommendation(output, {item.evidence_id: item})


@pytest.mark.parametrize(
    "action", [action for action in RecommendedNextAction if action != RecommendedNextAction.NO_RECOMMENDATION]
)
def test_absent_evidence_forces_abstention_for_every_review_action(action):
    output = recommendation(
        recommended_next_action=action, rationale=[{"statement": "Generic outside-knowledge review advice."}]
    )
    with pytest.raises(EvidenceValidationError, match="must abstain"):
        validate_recommendation(output, {})


def test_cited_claim_rationale_supports_review_action():
    item = EvidenceItem(evidence_id="claim:demo", source_type="CLAIM_DATA", source_reference="demo", summary="Fact")
    output = recommendation(
        rationale=[
            {"statement": "The collected claim data supports standard review.", "evidence_ids": [item.evidence_id]}
        ]
    )
    assert validate_recommendation(output, {item.evidence_id: item}) == []


def test_missing_evidence_can_be_explained_without_invented_source():
    output = recommendation(
        recommended_next_action="NO_RECOMMENDATION",
        rationale=[{"statement": "No tools returned usable evidence."}],
        uncertainties=["Stored assessments and retrieval are unavailable."],
    )
    assert validate_recommendation(output, {}) == []
    assert output.uncertainties and output.human_review_required is True
