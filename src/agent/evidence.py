"""Application-owned evidence provenance and authoritative citation resolution."""

from collections.abc import Mapping

from agent.models import AgentRecommendation, EvidenceItem, EvidenceSourceType, RecommendedNextAction
from knowledge.models import KnowledgeCitation


class EvidenceValidationError(ValueError):
    """The finalizer cited evidence outside the collected run context."""


def validate_recommendation(
    recommendation: AgentRecommendation,
    registry: Mapping[str, EvidenceItem],
) -> list[KnowledgeCitation]:
    """Validate every citation and resolve document metadata without trusting LLMs.

    A rationale must cite collected evidence when evidence is available. With an
    empty registry, only a recommendation about missing information is possible;
    its rationale can describe that absence without inventing a source.
    """
    if any(key != item.evidence_id for key, item in registry.items()):
        raise EvidenceValidationError("Evidence registry keys do not match evidence IDs")
    cited = list(recommendation.evidence_ids)
    for rationale in recommendation.rationale:
        if registry and not rationale.evidence_ids:
            raise EvidenceValidationError("A rationale must cite collected evidence")
        cited.extend(rationale.evidence_ids)
    cited = list(dict.fromkeys(cited))
    if set(cited) - registry.keys():
        raise EvidenceValidationError("Recommendation cited IDs absent from collected evidence")
    if recommendation.recommended_next_action != RecommendedNextAction.NO_RECOMMENDATION:
        if not registry:
            raise EvidenceValidationError("A recommendation without collected evidence must abstain")
        if not recommendation.rationale or not cited:
            raise EvidenceValidationError("A supported recommendation must include at least one cited rationale")

    citations = []
    knowledge_types = {EvidenceSourceType.POLICY_CHUNK, EvidenceSourceType.PROCEDURE_CHUNK}
    for evidence_id in cited:
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
