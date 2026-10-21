from agent.models import (
    AgentToolName,
    EvidenceItem,
    EvidenceSourceType,
)
from knowledge.models import KnowledgeFilter, KnowledgeSourceType, applicable_on
from retrieval.models import RetrievalUnavailableError

from ._data import MAX_KNOWLEDGE_EXCERPT_CHARS, MAX_VALUE_CHARS


def _knowledge(self, claim_id, tool, arguments):
    """
    Authoritative product/date cannot be set by the planner.

    Preserve exact authoritative citation fields within a finite context; oversized metadata is a
    source-contract failure.

    Ranking depends on the query; keep it in the observation rather than changing a stable chunk's
    evidence payload between searches.
    """
    claim = self._claim(claim_id)
    policy = tool == AgentToolName.SEARCH_POLICY
    source_type = KnowledgeSourceType.POLICY if policy else KnowledgeSourceType.PROCEDURE
    product = claim.policy.product if policy else None
    as_of_date = claim.incident.date if policy else None
    try:
        retriever = self.knowledge_retriever
        if retriever is None and self.knowledge_retriever_loader is not None:
            retriever = self.knowledge_retriever_loader()
            self.knowledge_retriever = retriever
        if retriever is None:
            return self._missing(tool, "Knowledge retrieval is not configured", unavailable=True)
        chunks = retriever.search(
            arguments.query,
            source_type,
            filters=KnowledgeFilter(product=product, as_of_date=as_of_date),
            limit=arguments.limit,
        )
    except RetrievalUnavailableError:
        return self._missing(tool, "Knowledge retrieval is unavailable", unavailable=True)
    evidence, summaries = _knowledge_evidence(chunks, source_type, product, as_of_date, arguments)
    if not evidence:
        return self._missing(tool, "No applicable knowledge excerpts were retrieved")
    return self._observation(tool, {"source_type": source_type.value, "results": summaries}, evidence)


def _knowledge_evidence(chunks, source_type, product, as_of_date, arguments):
    policy = source_type == KnowledgeSourceType.POLICY
    evidence, summaries, seen = [], [], set()
    for chunk in chunks:
        if (
            chunk.source_type != source_type
            or chunk.chunk_id in seen
            or (policy and chunk.product != product)
            or not applicable_on(chunk, as_of_date)
        ):
            continue
        seen.add(chunk.chunk_id)
        metadata = _knowledge_metadata(chunk)
        evidence.append(
            EvidenceItem(
                evidence_id=f"knowledge:{chunk.chunk_id}",
                source_type=EvidenceSourceType.POLICY_CHUNK if policy else EvidenceSourceType.PROCEDURE_CHUNK,
                source_reference=chunk.chunk_id,
                summary=chunk.text[:MAX_KNOWLEDGE_EXCERPT_CHARS],
                structured_data=metadata,
            )
        )
        summaries.append({**metadata, "similarity_score": chunk.similarity_score})
        if len(evidence) == arguments.limit:
            break
    return evidence, summaries


def _knowledge_metadata(chunk):
    metadata_limits = {
        "chunk_id": 250,
        "source_id": 300,
        "source_version": 100,
        "title": 500,
        "section": 500,
        "language": 16,
    }
    if any(len(getattr(chunk, field)) > maximum for field, maximum in metadata_limits.items()):
        raise ValueError("Knowledge citation metadata exceeds agent context limits")
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "source_version": chunk.source_version,
        "title": chunk.title,
        "section": chunk.section,
        "product": chunk.product[:MAX_VALUE_CHARS] if chunk.product else None,
        "language": chunk.language,
        "effective_from": str(chunk.effective_from) if chunk.effective_from else None,
        "effective_to": str(chunk.effective_to) if chunk.effective_to else None,
        "excerpt_truncated": len(chunk.text) > MAX_KNOWLEDGE_EXCERPT_CHARS,
    }
