"""Support."""

import pytest
from qdrant_client import QdrantClient

from retrieval.qdrant_store import QdrantClaimStore
from tests.infrastructure.support.retrieval_fakes import FakeEmbeddings


@pytest.fixture
def components():
    client = QdrantClient(":memory:")
    embeddings = FakeEmbeddings()
    store = QdrantClaimStore(client, "test_similar_v1", embeddings)
    try:
        yield embeddings, store
    finally:
        store.close()


def canonical_text():
    return "\n".join(
        [
            "claim_type: AUTO_COLLISION",
            "collision_type: FRONT_COLLISION",
            "location: Bordeaux",
            "vehicle_make: Renault",
            "vehicle_model: Clio",
            "vehicle_year: 2021",
            "declared_damage: FRONT_BUMPER, LEFT_HEADLIGHT",
            "injuries_declared: false",
            "repair_estimate_amount: 3160",
            "repair_estimate_currency: EUR",
        ]
    )


def assert_representation_allowlist(current, expected):
    forbidden = [
        "claim_id",
        "customer",
        "policy",
        "uuid",
        "triage",
        "workflow",
        "confidence",
        "anomaly",
        "investigation",
        "review_score",
        "priority",
        "model_version",
        "mlflow",
    ]
    assert all(word not in expected.lower() for word in forbidden)
    assert set(vars(current.facts)) == {
        "claim_type",
        "collision_type",
        "location",
        "vehicle_make",
        "vehicle_model",
        "vehicle_year",
        "declared_damage",
        "injuries_declared",
        "repair_amount",
        "repair_currency",
    }
