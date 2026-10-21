"""Authoritative markdown sources with strict JSON sidecars."""

import json
from dataclasses import dataclass
from pathlib import Path

from retrieval.knowledge.models import KnowledgeChunk, KnowledgeSource, KnowledgeSourceType
from retrieval.knowledge.sources.chunking import chunk_markdown

DEMO_NOTICE = "DEMO / SYNTHETIC DOCUMENT — NOT A REAL INSURANCE CONTRACT"


@dataclass(frozen=True)
class KnowledgeCatalog:
    sources: tuple[KnowledgeSource, ...]
    chunks: dict[str, KnowledgeChunk]

    @classmethod
    def load(cls, root: Path, max_chars: int = 600):
        """
        The sidecar cannot redirect the reader to another file.
        """
        sources, chunks, identities = [], {}, set()
        for path in sorted(root.rglob("*.md")):
            sidecar = path.with_suffix(".metadata.json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            source = KnowledgeSource(**metadata, file_path=path.relative_to(root).as_posix())
            expected_dir = "policies" if source.source_type == KnowledgeSourceType.POLICY else "procedures"
            if path.relative_to(root).parts[0] != expected_dir:
                raise ValueError(f"Wrong source directory for {source.source_id}")
            identity = (source.source_id, source.version)
            if identity in identities:
                raise ValueError(f"Duplicate source/version: {identity}")
            identities.add(identity)
            markdown = path.read_text(encoding="utf-8")
            if DEMO_NOTICE not in markdown:
                raise ValueError(f"Missing synthetic notice: {path.name}")
            source_chunks = chunk_markdown(source, markdown, max_chars)
            if not source_chunks:
                raise ValueError(f"Empty knowledge source: {path.name}")
            sources.append(source)
            chunks.update({c.chunk_id: c for c in source_chunks})
        if not sources:
            raise ValueError("No demo knowledge sources found")
        return cls(tuple(sources), chunks)
