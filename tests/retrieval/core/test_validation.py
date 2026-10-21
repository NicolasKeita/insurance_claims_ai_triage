"""Test validation."""

import pytest

from persistence.repositories import ClaimNotFoundError
from retrieval.config import RetrievalConfig
from retrieval.models import RetrievalUnavailableError
from retrieval.representation import (
    build_retrieval_document,
)
from retrieval.similar_claims import SimilarClaimsService
from tests.retrieval.core._support import components
from tests.retrieval.core._support import components as components
from tests.support.retrieval_fakes import FakeRepository, claim


@pytest.mark.parametrize("limit", [0, -1, 21, 1.5, True])
def test_invalid_limit_rejected_before_io(components, limit):
    embeddings, store = components
    with pytest.raises(ValueError, match="limit"):
        SimilarClaimsService(FakeRepository([]), embeddings, store).search("absent", limit)


def test_unknown_and_missing_index_are_explicit_no_automatic_creation(components):
    embeddings, store = components
    service = SimilarClaimsService(FakeRepository([claim()]), embeddings, store)
    with pytest.raises(ClaimNotFoundError):
        service.search("missing")
    with pytest.raises(RetrievalUnavailableError, match="index is missing"):
        service.search(claim().claim_id)
    assert not store.client.collection_exists(store.collection)


def test_invalid_vectors_rejected(components):
    _, store = components
    store.ensure_collection()
    document = build_retrieval_document(claim())
    for vector in ([0, 0, 0, 0], [float("nan"), 0, 0, 0], [1, 0], [2, 0, 0, 0]):
        with pytest.raises(ValueError, match="normalized"):
            store.upsert([document], [vector])


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("CLAIMS_INDEX_BATCH_SIZE", "16")
    monkeypatch.setenv("CLAIMS_SIMILAR_COLLECTION", "insurance_claims_similar_v2")
    assert RetrievalConfig.from_env().batch_size == 16
    assert RetrievalConfig.from_env().collection == "insurance_claims_similar_v2"
    with pytest.raises(ValueError):
        RetrievalConfig(batch_size=0)
