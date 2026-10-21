"""Test index."""

import json

import pytest
from qdrant_client import models

from retrieval.models import IndexCompatibilityError
from retrieval.representation import (
    CLAIM_RETRIEVAL_REPRESENTATION_VERSION,
)
from retrieval.similar_claims import SimilarClaimsIndexer
from tests.infrastructure.support.retrieval_fakes import FakeRepository, claim
from tests.intelligence.retrieval.core._support import components
from tests.intelligence.retrieval.core._support import components as components


def test_collection_upsert_idempotence_and_manifest(components, tmp_path):
    embeddings, store = components
    repository = FakeRepository([claim(), claim(2, days=1), claim(3, days=2)])
    indexer = SimilarClaimsIndexer(repository, embeddings, store, batch_size=2)
    for _ in range(2):
        stats = indexer.index_all()
        assert stats.claims_read == stats.claims_embedded == stats.points_upserted == 3
        assert store.count() == 3
    info = store.client.get_collection(store.collection)
    assert info.config.params.vectors.size == embeddings.dimension
    assert info.config.params.vectors.distance == models.Distance.COSINE
    assert info.config.metadata["claim_retrieval"] == store.contract()
    point = store.client.retrieve(store.collection, [str(claim().point_id)])[0]
    assert point.payload == {
        "claim_id": "DEMO-TEST-1",
        "claim_type": "AUTO_COLLISION",
        "incident_date": "2026-09-20",
        "incident_date_ordinal": claim().incident_date.toordinal(),
        "representation_version": CLAIM_RETRIEVAL_REPRESENTATION_VERSION,
    }
    path = tmp_path / "index_manifest.json"
    manifest = indexer.write_manifest(path, stats)
    assert json.loads(path.read_text()) == manifest
    assert manifest["indexed_claim_count"] == 3
    assert manifest["normalized"] is True
    assert manifest["embedding_model"] == embeddings.model_name
    assert indexer.upsert_claim("DEMO-TEST-1").points_upserted == 1
    assert store.count() == 3


@pytest.mark.parametrize("difference", ["dimension", "distance", "model", "representation", "missing_contract"])
def test_incompatible_collection_never_silently_deleted(components, difference):
    embeddings, store = components
    contract = store.contract()
    dimension, distance = embeddings.dimension, models.Distance.COSINE
    if difference == "dimension":
        dimension += 1
    if difference == "distance":
        distance = models.Distance.DOT
    if difference == "model":
        contract["embedding_model"] = "same-dimension-different-model"
    if difference == "representation":
        contract["representation_version"] = "old-v0"
    store.client.create_collection(
        store.collection,
        vectors_config=models.VectorParams(size=dimension, distance=distance),
        metadata={} if difference == "missing_contract" else {"claim_retrieval": contract},
    )
    with pytest.raises(IndexCompatibilityError, match="new collection/version"):
        store.ensure_collection()
    assert store.client.collection_exists(store.collection)
    store.ensure_collection(recreate=True)
    store.require_collection()
