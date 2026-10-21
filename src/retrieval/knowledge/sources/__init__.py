"""Markdown + strict JSON sidecars are authoritative; Qdrant is a derived copy."""

from retrieval.knowledge.sources.catalog import DEMO_NOTICE, KnowledgeCatalog
from retrieval.knowledge.sources.chunking import chunk_markdown

__all__ = ["DEMO_NOTICE", "KnowledgeCatalog", "chunk_markdown"]
