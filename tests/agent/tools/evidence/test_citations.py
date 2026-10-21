"""Test citations."""

import pytest

from agent.evidence import EvidenceValidationError, validate_recommendation
from agent.models import (
    EvidenceItem,
    RecommendationRationale,
)
from agent.tools import ReadOnlyAgentTools
from tests.agent.tools._evidence import chunk, plan, recommendation
from tests.agent.tools._repositories import configured
from tests.agent.tools._repositories import configured as configured


def test_citations_resolve_authoritative_metadata_from_rationale(configured):
    claim, factory, _, _ = configured

    class Retriever:
        def search(self, *_args, **_kwargs):
            return [chunk(claim)]

    item = (
        ReadOnlyAgentTools(factory, knowledge_retriever=Retriever())
        .execute(claim.claim_id, plan("SEARCH_POLICY", query="coverage"))
        .evidence[0]
    )
    output = recommendation(
        rationale=[
            RecommendationRationale(statement="The synthetic excerpt supports review.", evidence_ids=[item.evidence_id])
        ]
    )
    citations = validate_recommendation(output, {item.evidence_id: item})
    assert len(citations) == 1
    assert citations[0].model_dump() == {
        "chunk_id": "synthetic-policy-chunk",
        "source_id": "demo-policy",
        "source_version": "v1",
        "title": "Synthetic policy",
        "section": "Collision damage",
    }


@pytest.mark.parametrize("unknown_in_rationale", [False, True])
def test_invented_evidence_rejected_in_every_citation_field(unknown_in_rationale):
    output = recommendation(
        **(
            {"rationale": [{"statement": "Claim fact", "evidence_ids": ["claim:invented"]}]}
            if unknown_in_rationale
            else {"evidence_ids": ["claim:invented"]}
        )
    )
    with pytest.raises(EvidenceValidationError, match="absent"):
        validate_recommendation(output, {})


def test_available_evidence_requires_rationale_citations():
    item = EvidenceItem(evidence_id="claim:demo", source_type="CLAIM_DATA", source_reference="demo", summary="Fact")
    with pytest.raises(EvidenceValidationError, match="rationale"):
        validate_recommendation(recommendation(rationale=[{"statement": "Unsupported fact"}]), {item.evidence_id: item})
