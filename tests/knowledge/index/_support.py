"""Support."""

from datetime import date

import pytest
from qdrant_client import QdrantClient, models

from knowledge.config import KnowledgeConfig
from knowledge.models import KnowledgeFilter, KnowledgeSource
from knowledge.models import KnowledgeSourceType as Kind
from knowledge.qdrant_store import QdrantKnowledgeStore
from tests.support.knowledge_fakes import FakeEmbeddingProvider


@pytest.fixture
def store():
    store = QdrantKnowledgeStore(QdrantClient(":memory:"), FakeEmbeddingProvider(), KnowledgeConfig())
    try:
        yield store
    finally:
        store.close()


def source(**updates):
    data = dict(
        source_id="TEST",
        source_type="POLICY",
        title="Demo",
        version="1",
        product="AUTO_PREMIUM",
        language="en",
        file_path="policies/test.md",
        effective_from=date(2026, 1, 1),
        effective_to=None,
    )
    return KnowledgeSource(**(data | updates))


def assert_authoritative_chunks(store, service, target):
    store.client.set_payload(
        store.config.policy_collection, payload={"text": "FORGED", "title": "FORGED"}, points=[target.chunk_id]
    )
    resolved = service.search(target.text, Kind.POLICY, filters=KnowledgeFilter(product="AUTO_PREMIUM"), limit=1)[0]
    assert resolved.text == target.text and resolved.title == target.title
    from uuid import uuid4

    foreign_id = str(uuid4())
    store.client.upsert(
        store.config.policy_collection,
        points=[
            models.PointStruct(
                id=foreign_id,
                vector=store.embeddings.embed_query(target.text),
                payload=target.model_dump(mode="json") | {"chunk_id": foreign_id},
            )
        ],
    )
    assert len(service.search(target.text, Kind.POLICY, filters=KnowledgeFilter(product="AUTO_PREMIUM"), limit=5)) == 5
