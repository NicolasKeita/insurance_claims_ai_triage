"""Test retrieval."""

from datetime import date

import pytest

from retrieval.knowledge.indexing import KnowledgeIndexer
from retrieval.knowledge.models import KnowledgeFilter, applicable_on
from retrieval.knowledge.models import KnowledgeSourceType as Kind
from retrieval.knowledge.retrieval import KnowledgeRetriever
from retrieval.knowledge.sources import KnowledgeCatalog, chunk_markdown
from tests.intelligence.knowledge.index._support import assert_authoritative_chunks, source, store
from tests.intelligence.knowledge.index._support import store as store


def test_retrieval_filters_limit_rank_authoritative_text_and_stale_chunks(store, tmp_path):
    """
    Do not expose forged payload text or provenance.

    A foreign ID cannot resolve to a local chunk; refill to requested limit.
    """
    KnowledgeIndexer(store).index_all(tmp_path / "manifest.json")
    service = KnowledgeRetriever(store)
    catalog = KnowledgeCatalog.load(store.config.knowledge_dir)
    target = next(
        c for c in catalog.chunks.values() if c.source_id == "AUTO_PREMIUM_POLICY" and c.section == "Deductible"
    )
    response = service.search(
        target.text, Kind.POLICY, filters=KnowledgeFilter(product="AUTO_PREMIUM", language="en"), limit=3
    )
    assert len(response) == 3 and response[0].chunk_id == target.chunk_id
    assert [r.similarity_score for r in response] == sorted((r.similarity_score for r in response), reverse=True)
    assert all(r.product == "AUTO_PREMIUM" and r.source_type == Kind.POLICY for r in response)
    assert service.search("test", Kind.POLICY, filters=KnowledgeFilter(language="fr")) == []
    assert service.search("test", Kind.POLICY, filters=KnowledgeFilter(product="NONEXISTENT")) == []
    selected = service.search("test", Kind.POLICY, filters=KnowledgeFilter(source_id="AUTO_BASIC_POLICY"))
    assert selected and all(r.source_id == "AUTO_BASIC_POLICY" for r in selected)
    both = service.search(
        "test", Kind.POLICY, filters=KnowledgeFilter(product="AUTO_PREMIUM", source_id="AUTO_BASIC_POLICY")
    )
    assert both == []
    assert service.search("test", Kind.POLICY, filters=KnowledgeFilter(as_of_date=date(2025, 12, 31))) == []
    assert all(r.source_type == Kind.PROCEDURE and r.product is None for r in service.search("test", Kind.PROCEDURE))
    assert_authoritative_chunks(store, service, target)


def test_temporal_pure_and_qdrant_boundaries(store):
    bounded = source(effective_to=date(2026, 10, 1))
    assert not applicable_on(bounded, date(2025, 12, 31))
    assert applicable_on(bounded, date(2026, 1, 1))
    assert applicable_on(bounded, date(2026, 9, 30))
    assert not applicable_on(bounded, date(2026, 10, 1))
    assert applicable_on(bounded, None)
    assert applicable_on(source(effective_from=None), date(1900, 1, 1))
    with pytest.raises(ValueError, match="exclusive"):
        source(effective_to=date(2026, 1, 1))
    chunks = []
    for s in (
        bounded,
        source(version="2", effective_from=date(2026, 10, 1)),
        source(source_id="TIMELESS", effective_from=None),
    ):
        chunks.extend(chunk_markdown(s, "## Rules\n\nA demo rule."))
    store.ensure_collection(Kind.POLICY)
    store.upsert(Kind.POLICY, chunks, store.embeddings.embed_documents([c.text for c in chunks]))
    for day, expected in (
        (date(2026, 1, 1), {chunks[0].chunk_id, chunks[2].chunk_id}),
        (date(2026, 10, 1), {chunks[1].chunk_id, chunks[2].chunk_id}),
        (date(2025, 1, 1), {chunks[2].chunk_id}),
    ):
        hits, _ = store.query(
            Kind.POLICY, store.embeddings.embed_query("rule"), filters=KnowledgeFilter(as_of_date=day), limit=20
        )
        assert {i for i, _ in hits} == expected
