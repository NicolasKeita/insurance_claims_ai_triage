"""Application-owned evidence provenance and authoritative citation resolution."""

from collections.abc import Mapping

from agent.models import (
    AgentRecommendation,
    EvidenceItem,
    EvidenceSourceType,
    RecommendedNextAction,
)
from agent.review_policy import REPAIR_FOLLOW_UP_ACTIONS, repair_discrepancy_sources
from retrieval.knowledge.models import KnowledgeCitation


class EvidenceValidationError(ValueError):
    """The finalizer violated evidence provenance or a required follow-up."""


def rationale_evidence_ids(recommendation: AgentRecommendation) -> list[str]:
    """Derive the source union in first-citation order, independently of LLM bookkeeping."""
    return list(dict.fromkeys(eid for rationale in recommendation.rationale for eid in rationale.evidence_ids))


def _validate_follow_up(recommendation, registry, supporting_ids):
    sources = repair_discrepancy_sources(registry)
    if not sources:
        return
    if recommendation.recommended_next_action not in REPAIR_FOLLOW_UP_ACTIONS:
        raise EvidenceValidationError("Unexplained repair quote items require a follow-up action")
    if not set(sources).intersection(supporting_ids):
        raise EvidenceValidationError("Follow-up rationale must cite the collected repair assessment")


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
        raise EvidenceValidationError("Evidence registry keys do not match evidence IDs")
    supporting_ids = rationale_evidence_ids(recommendation)
    for rationale in recommendation.rationale:
        if registry and not rationale.evidence_ids:
            raise EvidenceValidationError("A rationale must cite collected evidence")
    if set([*recommendation.evidence_ids, *supporting_ids]) - registry.keys():
        raise EvidenceValidationError("Recommendation cited IDs absent from collected evidence")
    if recommendation.recommended_next_action != RecommendedNextAction.NO_RECOMMENDATION:
        if not registry:
            raise EvidenceValidationError("A recommendation without collected evidence must abstain")
        if not recommendation.rationale or not supporting_ids:
            raise EvidenceValidationError("A supported recommendation must include at least one cited rationale")
    _validate_follow_up(recommendation, registry, supporting_ids)

    citations = []
    knowledge_types = {EvidenceSourceType.POLICY_CHUNK, EvidenceSourceType.PROCEDURE_CHUNK}
    for evidence_id in supporting_ids:
        item = registry[evidence_id]
        if item.source_type not in knowledge_types:
            continue
        metadata = item.structured_data or {}
        fields = ("chunk_id", "source_id", "source_version", "title", "section")
        if any(not isinstance(metadata.get(field), str) or not metadata[field] for field in fields):
            raise EvidenceValidationError("Knowledge evidence is missing authoritative citation metadata")
        if metadata["chunk_id"] != item.source_reference:
            raise EvidenceValidationError("Knowledge evidence reference does not match its chunk ID")
        citations.append(KnowledgeCitation(**{field: metadata[field] for field in fields}))
    return citations
