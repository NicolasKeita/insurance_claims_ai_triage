"""Real test Qdrant connections, owning only collections generated here."""

import os
import re
from contextlib import contextmanager
from uuid import uuid4

from integration_support import get_test_qdrant_url
from qdrant_client import QdrantClient


@contextmanager
def temporary_qdrant_collections(*labels: str):
    """Close every connection and attempt every reserved collection's cleanup."""
    if not labels or any(re.fullmatch(r"[a-z][a-z0-9_]*", label) is None for label in labels):
        raise ValueError("Test collection labels must contain only lowercase letters, digits and underscores")
    suffix = uuid4().hex
    collections = tuple(f"test_{label}_{suffix}" for label in labels)
    url = get_test_qdrant_url()
    client = QdrantClient(url=url, api_key=os.environ.get("TEST_QDRANT_API_KEY"), timeout=5, check_compatibility=False)
    reserved = False
    try:
        try:
            existing = {collection.name for collection in client.get_collections().collections}
        except Exception as error:
            raise RuntimeError(
                f"Test Qdrant is unreachable at {url}. Start it with "
                "'docker compose --profile test up -d --wait qdrant-test' or configure "
                "TEST_QDRANT_URL (and TEST_QDRANT_API_KEY in CI). "
                f"Original error: {type(error).__name__}: {error}"
            ) from error
        if existing.intersection(collections):
            raise RuntimeError("Refusing to reserve an existing Qdrant collection; no collection was changed")
        reserved = True
        yield client, collections
    finally:
        errors = []
        if reserved:
            for collection in collections:
                try:
                    if client.collection_exists(collection):
                        client.delete_collection(collection)
                except Exception as error:
                    errors.append(RuntimeError(f"Could not clean up test Qdrant collection {collection}: {error}"))
        try:
            client.close()
        except Exception as error:
            errors.append(error)
        if errors:
            raise ExceptionGroup("Test Qdrant cleanup failed; inspect the reserved test collections", errors)
