"""Batch indexing of both knowledge collections with a manifest."""

import json
from datetime import datetime, timezone
from pathlib import Path

from knowledge.models import KnowledgeSourceType
from knowledge.qdrant_store import QdrantKnowledgeStore
from knowledge.sources import KnowledgeCatalog
from retrieval.models import IndexCompatibilityError


class KnowledgeIndexer:
    def __init__(self, store: QdrantKnowledgeStore):
        self.store = store

    def index_all(self, manifest_path: Path, *, recreate: bool = False) -> dict:
        """
        Parse the entire corpus and resolve embedding configuration before mutation.

        Validate both existing collections before upserting either one.
        """
        config = self.store.config
        catalog = KnowledgeCatalog.load(config.knowledge_dir, config.max_chunk_chars)
        self._validate_existing(config, recreate)
        collections = [self._index_kind(kind, catalog, config, recreate) for kind in KnowledgeSourceType]
        manifest = {"manifest_version": "knowledge_index_manifest_v1", "collections": collections}
        _write_manifest(manifest_path, manifest)
        return manifest

    def _validate_existing(self, config, recreate):
        """Resolve contracts, and check ownership before any destructive action."""
        for kind in KnowledgeSourceType:
            self.store.contract(kind)
        if recreate:
            for kind in KnowledgeSourceType:
                self._assert_owned(config, kind)
            return
        for kind in KnowledgeSourceType:
            if self.store.client.collection_exists(config.collection(kind)):
                self.store.require_collection(kind)

    def _assert_owned(self, config, kind):
        if not self.store.client.collection_exists(config.collection(kind)):
            return
        info = self.store.client.get_collection(config.collection(kind))
        owned = (info.config.metadata or {}).get("knowledge_rag")
        if not isinstance(owned, dict) or owned.get("source_type") != kind.value:
            raise IndexCompatibilityError("Refusing to recreate a collection not owned by this RAG source type")

    def _index_kind(self, kind, catalog, config, recreate):
        """Upsert one source type in batches and describe the resulting collection."""
        self.store.ensure_collection(kind, recreate=recreate)
        chunks = [c for c in catalog.chunks.values() if c.source_type == kind]
        for start in range(0, len(chunks), config.batch_size):
            batch = chunks[start : start + config.batch_size]
            vectors = self.store.embeddings.embed_documents([c.text for c in batch])
            self.store.upsert(kind, batch, vectors)
        sources = [s for s in catalog.sources if s.source_type == kind]
        return {
            "collection": config.collection(kind),
            **self.store.contract(kind),
            "number_of_sources": len(sources),
            "number_of_chunks": len(chunks),
            "point_count": self.store.count(kind),
            "indexed_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_versions": [
                {
                    "source_id": s.source_id,
                    "version": s.version,
                    "file_path": s.file_path,
                    "chunk_ids": [
                        c.chunk_id for c in chunks if c.source_id == s.source_id and c.source_version == s.version
                    ],
                }
                for s in sources
            ],
        }


def _write_manifest(manifest_path, manifest):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
