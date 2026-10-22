"""Final source lists derive from cited rationale, after validating all generated IDs."""

import pytest

from agent.models import EvidenceItem, RecommendedNextAction, ToolObservation
from agent.service import AgentRunError
from tests.agent.graph._support import service
from tests.agent.safety._constants import CLAIM_ID
from tests.infrastructure.support.agent_fakes import FakeAgentTools, FakeRunStore, make_recommendation, make_step

CLAIM_SOURCE = f"claim:{CLAIM_ID}"
USED_POLICY = "knowledge:used-policy"
UNUSED_POLICY = "knowledge:unused-policy"


def policy_evidence(identifier):
    chunk_id = identifier.removeprefix("knowledge:")
    return EvidenceItem(
        evidence_id=identifier,
        source_type="POLICY_CHUNK",
        source_reference=chunk_id,
        summary="Synthetic collision review guidance.",
        structured_data={
            "chunk_id": chunk_id,
            "source_id": "demo-policy",
            "source_version": "1",
            "title": "Demo policy",
            "section": chunk_id,
        },
    )


def review(recommendation, store=None):
    tools = FakeAgentTools(
        observations={
            "SEARCH_POLICY": ToolObservation(
                tool_name="SEARCH_POLICY",
                status="OK",
                evidence=[policy_evidence(USED_POLICY), policy_evidence(UNUSED_POLICY)],
            )
        }
    )
    steps = ["GET_CLAIM", make_step("SEARCH_POLICY", {"query": "collision review"}), "FINISH"]
    agent, _, _ = service(steps, recommendation, tools=tools, store=store)
    return agent.review_claim(CLAIM_ID)


@pytest.mark.parametrize("declared", [[], [CLAIM_SOURCE], [UNUSED_POLICY, USED_POLICY, CLAIM_SOURCE, USED_POLICY]])
def test_completed_recommendation_and_saved_citations_follow_only_rationale(declared):
    recommendation = make_recommendation([CLAIM_SOURCE, USED_POLICY])
    recommendation.rationale.append(recommendation.rationale[0].model_copy(deep=True))
    recommendation.evidence_ids = declared
    store = FakeRunStore()
    result = review(recommendation, store)
    assert result.status == "COMPLETED"
    assert result.recommendation.evidence_ids == [CLAIM_SOURCE, USED_POLICY]
    assert [citation.chunk_id for citation in result.recommendation.citations] == ["used-policy"]
    assert result.recommendation.rationale == recommendation.rationale
    assert result.recommendation.case_summary == recommendation.case_summary
    assert result.recommendation.recommended_next_action == recommendation.recommended_next_action
    assert UNUSED_POLICY in {item.evidence_id for item in result.evidence}
    assert store.results == [result]


@pytest.mark.parametrize("field", ["evidence_ids", "rationale"])
def test_normalization_never_discards_invented_ids_before_validation(field):
    recommendation = make_recommendation([CLAIM_SOURCE, USED_POLICY])
    target = recommendation.evidence_ids if field == "evidence_ids" else recommendation.rationale[0].evidence_ids
    target.append("knowledge:invented")
    store = FakeRunStore()
    with pytest.raises(AgentRunError) as caught:
        review(recommendation, store)
    assert caught.value.failure_kind == "GROUNDING"
    assert store.results[0].status == "FAILED"
    assert store.results[0].recommendation is None


def test_abstention_without_rationale_does_not_publish_unused_collected_sources():
    recommendation = make_recommendation([])
    recommendation.recommended_next_action = RecommendedNextAction.NO_RECOMMENDATION
    recommendation.evidence_ids = [CLAIM_SOURCE, USED_POLICY]
    result = review(recommendation)
    assert result.recommendation.evidence_ids == []
    assert result.recommendation.citations == []
    assert len(result.evidence) == 3
