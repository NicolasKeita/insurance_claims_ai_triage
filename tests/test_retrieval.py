import json
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models

from persistence.repositories import ClaimNotFoundError
from retrieval.config import RetrievalConfig
from retrieval.embeddings import SentenceTransformerEmbeddingProvider
from retrieval.models import IndexCompatibilityError, RetrievalUnavailableError
from retrieval.qdrant_store import QdrantClaimStore
from retrieval.representation import (
    CLAIM_RETRIEVAL_REPRESENTATION_VERSION, build_claim_retrieval_text, build_retrieval_document,
)
from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService
from retrieval_fakes import FakeEmbeddings, FakeRepository, claim


@pytest.fixture
def components():
    client = QdrantClient(":memory:")
    embeddings = FakeEmbeddings()
    store = QdrantClaimStore(client, "test_similar_v1", embeddings)
    try:
        yield embeddings, store
    finally:
        store.close()


def test_canonical_representation_is_stable_and_allowlisted():
    current = claim()
    expected = "\n".join([
        "claim_type: AUTO_COLLISION", "collision_type: FRONT_COLLISION", "location: Bordeaux",
        "vehicle_make: Renault", "vehicle_model: Clio", "vehicle_year: 2021",
        "declared_damage: FRONT_BUMPER, LEFT_HEADLIGHT", "injuries_declared: false",
        "repair_estimate_amount: 3160", "repair_estimate_currency: EUR",
    ])
    assert build_claim_retrieval_text(current.facts) == expected
    reordered = replace(current.facts, declared_damage=tuple(reversed(current.facts.declared_damage)),
                        repair_amount=Decimal("3160.000000"), location=" Bordeaux\n ")
    assert build_claim_retrieval_text(reordered) == expected
    changed_identity = replace(current, claim_id="PRIVATE-ID", point_id=claim(9).point_id,
                               incident_date=current.incident_date - timedelta(days=10))
    assert build_retrieval_document(changed_identity).canonical_text == expected
    document = build_retrieval_document(current)
    assert document.representation_version == "claim_retrieval_v1"
    assert document.point_id == current.point_id
    forbidden = [
        "claim_id", "customer", "policy", "uuid", "triage", "workflow", "confidence",
        "anomaly", "investigation", "review_score", "priority", "model_version", "mlflow",
    ]
    assert all(word not in expected.lower() for word in forbidden)
    # The input type cannot carry identifiers or model outputs at all.
    assert set(vars(current.facts)) == {
        "claim_type", "collision_type", "location", "vehicle_make", "vehicle_model", "vehicle_year",
        "declared_damage", "injuries_declared", "repair_amount", "repair_currency",
    }


def test_import_api_and_provider_construction_loads_no_sentence_transformer():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import api.main; from retrieval.embeddings import SentenceTransformerEmbeddingProvider; "
        "p = SentenceTransformerEmbeddingProvider('unused'); "
        "assert p._model is None; assert 'sentence_transformers' not in sys.modules",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_provider_loads_once_and_normalizes_symmetric_embeddings(monkeypatch):
    loads, calls = [], []

    class Model:
        def __init__(self, name):
            loads.append(name)

        def get_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return [[1, 0, 0, 0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    provider = SentenceTransformerEmbeddingProvider("mock-model")
    assert loads == []
    assert provider.dimension == 4
    assert provider.embed_documents(["a", "b"]) == [[1, 0, 0, 0], [1, 0, 0, 0]]
    assert provider.embed_query("a") == [1, 0, 0, 0]
    assert provider.embed_documents([]) == []
    assert loads == ["mock-model"]
    assert all(call["normalize_embeddings"] for call in calls)


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
        "claim_id": "DEMO-TEST-1", "claim_type": "AUTO_COLLISION", "incident_date": "2026-09-20",
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
        store.collection, vectors_config=models.VectorParams(size=dimension, distance=distance),
        metadata={} if difference == "missing_contract" else {"claim_retrieval": contract},
    )
    with pytest.raises(IndexCompatibilityError, match="new collection/version"):
        store.ensure_collection()
    assert store.client.collection_exists(store.collection)
    store.ensure_collection(recreate=True)
    store.require_collection()


def test_ranking_top_k_and_strict_date_and_type_filters(components):
    embeddings, store = components
    claims = [
        claim(), claim(2, days=2), claim(3, days=4, amount="3500"),
        claim(4, days=6, collision="PARKING_DAMAGE"),
        claim(5, days=0), claim(6, days=-1), claim(7, days=10, claim_type="HOME"),
    ]
    repository = FakeRepository(claims)
    SimilarClaimsIndexer(repository, embeddings, store).index_all()
    service = SimilarClaimsService(repository, embeddings, store)
    results = service.search(claims[0].claim_id, 2).results
    assert [r.claim_id for r in results] == [claims[1].claim_id, claims[2].claim_id]
    assert results[0].similarity_score > results[1].similarity_score
    results = service.search(claims[0].claim_id, 20).results
    assert [r.claim_id for r in results] == [c.claim_id for c in claims[1:4]]
    assert all(r.incident_date < claims[0].incident_date for r in results)
    # Qdrant itself excludes type/date/current; do not rely on app filtering alone.
    page = store.query(embeddings.embed_query(build_retrieval_document(claims[0]).canonical_text),
                       claims[0].incident_date, "AUTO_COLLISION", 20)
    assert {h.point_id for h in page.hits} == {c.point_id for c in claims[1:4]}
    other_domain = service.search(claims[-1].claim_id).results
    assert other_domain == []  # compares to actual HOME, never hardcodes AUTO_COLLISION
    unfiltered = SimilarClaimsService(repository, embeddings, store, same_claim_type=False)
    assert claims[-1].claim_id in {r.claim_id for r in unfiltered.search(claims[0].claim_id).results}


def test_stale_points_ignored_and_postgres_facts_used(components):
    embeddings, store = components
    claims = [claim(), *[claim(i, days=i) for i in range(2, 7)]]
    repository = FakeRepository(claims)
    SimilarClaimsIndexer(repository, embeddings, store).index_all()
    del repository.claims[claims[1].point_id]
    repository.claims[claims[2].point_id] = replace(claims[2], incident_date=claims[0].incident_date)
    repository.claims[claims[3].point_id] = replace(claims[3], incident_date=claims[0].incident_date + timedelta(days=1))
    repository.claims[claims[4].point_id] = replace(claims[4], facts=replace(claims[4].facts, claim_type="HOME"))
    repository.claims[claims[5].point_id] = replace(claims[5], facts=replace(
        claims[5].facts, repair_amount=Decimal("3200"), collision_type="SIDE_COLLISION"))
    results = SimilarClaimsService(repository, embeddings, store).search(claims[0].claim_id).results
    assert len(results) == 1
    assert results[0].claim_id == claims[5].claim_id
    assert results[0].repair_amount == Decimal("3200")
    assert results[0].collision_type == "SIDE_COLLISION"


def test_search_refills_after_a_whole_page_of_deleted_points(components):
    embeddings, store = components
    repository = FakeRepository([claim(), *[claim(i, days=1) for i in range(2, 55)]])
    SimilarClaimsIndexer(repository, embeddings, store).index_all()
    # Keep only the last ranked candidate; determine ranking before removing rows.
    current = repository.get("DEMO-TEST-1")
    vector = embeddings.embed_query(build_retrieval_document(current).canonical_text)
    ordered = store.query(vector, current.incident_date, current.facts.claim_type, 100).hits
    last = repository.claims[ordered[-1].point_id]
    repository.claims = {current.point_id: current, last.point_id: last}
    response = SimilarClaimsService(repository, embeddings, store).search(current.claim_id, 1)
    assert [r.claim_id for r in response.results] == [last.claim_id]
    assert len(repository.bulk_reads) == 3


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
