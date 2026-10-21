"""Evidence."""

from agent.models import (
    AgentPlanStep,
    AgentRecommendation,
)
from retrieval.knowledge.models import RetrievedKnowledgeChunk


def plan(action, **arguments):
    return AgentPlanStep(action=action, reason="Collect relevant read-only evidence.", arguments=arguments)


def chunk(claim, **changes):
    return RetrievedKnowledgeChunk(
        **{
            "chunk_id": "synthetic-policy-chunk",
            "source_id": "demo-policy",
            "source_type": "POLICY",
            "source_version": "v1",
            "title": "Synthetic policy",
            "section": "Collision damage",
            "chunk_index": 0,
            "text": "Synthetic demonstration evidence.",
            "product": claim.policy.product,
            "language": "en",
            "effective_from": None,
            "effective_to": None,
            "chunking_version": "knowledge_chunk_v1",
            "similarity_score": 0.9,
            **changes,
        }
    )


def recommendation(**changes):
    return AgentRecommendation(
        **{
            "case_summary": "Review the available evidence.",
            "rationale": [],
            "uncertainties": [],
            "evidence_ids": [],
            "recommended_next_action": "CONTINUE_STANDARD_REVIEW",
            **changes,
        }
    )
