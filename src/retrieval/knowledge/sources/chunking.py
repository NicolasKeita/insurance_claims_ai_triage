"""Deterministic markdown chunking with stable UUID5 identities."""

import json
import re
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from retrieval.knowledge.config import KNOWLEDGE_CHUNKING_VERSION
from retrieval.knowledge.models import KnowledgeChunk, KnowledgeSource


def _split_long(text: str, maximum: int) -> list[str]:
    """
    Prefer sentence ends, then whitespace; hard split only a single oversized token.
    """
    pieces = []
    while len(text) > maximum:
        window = text[: maximum + 1]
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
    """
    H1 is document title; nested subsections retain their heading path.
    """
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    chunks = []
    for section, text in _sections(markdown, source.title):
        for block in _blocks(text, max_chars):
            chunks.append(_chunk(source, section, len(chunks), block, max_chars))
    return chunks


def _sections(markdown: str, title: str) -> list[tuple[str, str]]:
    """Split markdown into (heading path, paragraph text) sections."""
    section = title
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
            level, heading_title = len(heading[1]), heading[2]
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = heading_title
            section = " / ".join(v for k, v in sorted(headings.items()) if k > 1) or heading_title
        else:
            paragraphs.append(line)
    flush()
    return sections


def _blocks(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs into blocks that respect the maximum character budget."""
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
    return blocks


def _chunk(source, section, index, block, max_chars):
    """Build one chunk whose identity depends only on its source and content."""
    identity = json.dumps(
        [source.model_dump(mode="json"), KNOWLEDGE_CHUNKING_VERSION, max_chars, section, index, block],
        sort_keys=True,
        ensure_ascii=False,
    )
    chunk_id = str(uuid5(NAMESPACE_URL, sha256(identity.encode("utf-8")).hexdigest()))
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_id=source.source_id,
        source_type=source.source_type,
        source_version=source.version,
        title=source.title,
        section=section,
        chunk_index=index,
        text=block,
        product=source.product,
        language=source.language,
        effective_from=source.effective_from,
        effective_to=source.effective_to,
        chunking_version=KNOWLEDGE_CHUNKING_VERSION,
    )
