"""Markdown + strict JSON sidecars are authoritative; Qdrant is a derived copy."""

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from knowledge.config import KNOWLEDGE_CHUNKING_VERSION
from knowledge.models import KnowledgeChunk, KnowledgeSource, KnowledgeSourceType

DEMO_NOTICE = "DEMO / SYNTHETIC DOCUMENT — NOT A REAL INSURANCE CONTRACT"


def _split_long(text: str, maximum: int) -> list[str]:
    # Prefer sentence ends, then whitespace; hard split only a single oversized token.
    pieces = []
    while len(text) > maximum:
        window = text[:maximum + 1]
        boundaries = [m.end() for m in re.finditer(r"[.!?](?=\s)", window)]
        cut = boundaries[-1] if boundaries else window.rfind(" ", 0, maximum + 1)
        if cut < 1:
            cut = maximum
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def chunk_markdown(source: KnowledgeSource, markdown: str, max_chars: int = 600) -> list[KnowledgeChunk]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    section = source.title
    headings: dict[int, str] = {}
    paragraphs: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush():
        text = "\n".join(paragraphs).strip()
        if text:
            sections.append((section, text))
        paragraphs.clear()

    for line in markdown.replace("\r\n", "\n").splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level, title = len(heading[1]), heading[2]
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = title
            # H1 is document title; nested subsections retain their heading path.
            section = " / ".join(v for k, v in sorted(headings.items()) if k > 1) or title
        else:
            paragraphs.append(line)
    flush()

    chunks = []
    for section, text in sections:
        buffer = ""
        blocks = []
        for paragraph in re.split(r"\n\s*\n", text):
            for piece in _split_long(paragraph.strip(), max_chars):
                candidate = f"{buffer}\n\n{piece}" if buffer else piece
                if len(candidate) > max_chars:
                    blocks.append(buffer)
                    buffer = piece
                else:
                    buffer = candidate
        if buffer:
            blocks.append(buffer)
        for block in blocks:
            index = len(chunks)
            identity = json.dumps([source.model_dump(mode="json"), KNOWLEDGE_CHUNKING_VERSION,
                                   max_chars, section, index, block], sort_keys=True, ensure_ascii=False)
            chunk_id = str(uuid5(NAMESPACE_URL, sha256(identity.encode("utf-8")).hexdigest()))
            chunks.append(KnowledgeChunk(
                chunk_id=chunk_id, source_id=source.source_id, source_type=source.source_type,
                source_version=source.version, title=source.title, section=section,
                chunk_index=index, text=block, product=source.product, language=source.language,
                effective_from=source.effective_from, effective_to=source.effective_to,
                chunking_version=KNOWLEDGE_CHUNKING_VERSION,
            ))
    return chunks


@dataclass(frozen=True)
class KnowledgeCatalog:
    sources: tuple[KnowledgeSource, ...]
    chunks: dict[str, KnowledgeChunk]

    @classmethod
    def load(cls, root: Path, max_chars: int = 600):
        sources, chunks, identities = [], {}, set()
        for path in sorted(root.rglob("*.md")):
            sidecar = path.with_suffix(".metadata.json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            # The sidecar cannot redirect the reader to another file.
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
