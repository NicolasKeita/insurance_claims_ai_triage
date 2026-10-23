"""Application-owned evidence provenance and authoritative citation resolution."""

from collections.abc import Mapping

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.diagnostics import safe_code
from agent.models import (
    AgentRecommendation,
    EvidenceItem,
    EvidenceSourceType,
    RecommendedNextAction,
)
from agent.review_policy import FOLLOW_UP_ACTIONS, follow_up_sources
from retrieval.knowledge.models import KnowledgeCitation


class EvidenceValidationError(ValueError):
    """The finalizer violated evidence provenance or a required follow-up."""

    def __init__(self, message, *, code="INVALID_EVIDENCE"):
        super().__init__(message)
        self.code = safe_code(code)


def rationale_evidence_ids(recommendation: AgentRecommendation) -> list[str]:
    """Derive the source union in first-citation order, independently of LLM bookkeeping."""
    return list(dict.fromkeys(eid for rationale in recommendation.rationale for eid in rationale.evidence_ids))


def validate_recommendation_versions(recommendation):
    if (recommendation.agent_version, recommendation.prompt_version) != (
        CLAIMS_AGENT_VERSION,
        CLAIMS_AGENT_PROMPT_VERSION,
    ):
        raise EvidenceValidationError("Recommendation versions do not match this agent", code="VERSION_MISMATCH")


def _validate_follow_up(recommendation, registry, supporting_ids):
    sources = follow_up_sources(registry)
    if not sources:
        return
    if recommendation.recommended_next_action not in FOLLOW_UP_ACTIONS:
        raise EvidenceValidationError(
            "Recorded investigation signals require a follow-up action", code="INCOMPATIBLE_FOLLOW_UP"
        )
    if not set(sources).intersection(supporting_ids):
        raise EvidenceValidationError(
            "Follow-up rationale must cite the collected investigation assessment",
            code="FOLLOW_UP_ASSESSMENT_NOT_CITED",
        )


def validate_recommendation(
    recommendation: AgentRecommendation,
    registry: Mapping[str, EvidenceItem],
) -> list[KnowledgeCitation]:
    """Validate all supplied IDs; resolve metadata for sources supporting rationale.

    A rationale must cite collected evidence when evidence is available. With an
    empty registry, only a recommendation about missing information is possible;
    its rationale can describe that absence without inventing a source.
    Generated top-level IDs are checked even when normalization will discard them.
    """
    if any(key != item.evidence_id for key, item in registry.items()):
        raise EvidenceValidationError("Evidence registry keys do not match evidence IDs", code="INVALID_REGISTRY")
    supporting_ids = rationale_evidence_ids(recommendation)
    for rationale in recommendation.rationale:
        if registry and not rationale.evidence_ids:
            raise EvidenceValidationError("A rationale must cite collected evidence", code="UNCITED_RATIONALE")
    if set([*recommendation.evidence_ids, *supporting_ids]) - registry.keys():
        raise EvidenceValidationError(
            "Recommendation cited IDs absent from collected evidence", code="UNKNOWN_EVIDENCE_ID"
        )
    if recommendation.recommended_next_action != RecommendedNextAction.NO_RECOMMENDATION:
        if not registry:
            raise EvidenceValidationError(
                "A recommendation without collected evidence must abstain", code="NO_EVIDENCE"
            )
        if not recommendation.rationale or not supporting_ids:
            raise EvidenceValidationError(
                "A supported recommendation must include at least one cited rationale", code="NO_SUPPORTING_RATIONALE"
            )
    _validate_follow_up(recommendation, registry, supporting_ids)
    return _knowledge_citations(registry, supporting_ids)


def _knowledge_citations(registry, supporting_ids):
    citations = []
    knowledge_types = {EvidenceSourceType.POLICY_CHUNK, EvidenceSourceType.PROCEDURE_CHUNK}
    for evidence_id in supporting_ids:
        item = registry[evidence_id]
        if item.source_type not in knowledge_types:
            continue
        metadata = item.structured_data or {}
        fields = ("chunk_id", "source_id", "source_version", "title", "section")
        if any(not isinstance(metadata.get(field), str) or not metadata[field] for field in fields):
            raise EvidenceValidationError(
                "Knowledge evidence is missing authoritative citation metadata", code="INVALID_KNOWLEDGE_METADATA"
            )
        if metadata["chunk_id"] != item.source_reference:
            raise EvidenceValidationError(
                "Knowledge evidence reference does not match its chunk ID", code="KNOWLEDGE_REFERENCE_MISMATCH"
            )
        citations.append(KnowledgeCitation(**{field: metadata[field] for field in fields}))
    return citations
