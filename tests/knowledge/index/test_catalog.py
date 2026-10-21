"""Test catalog."""

from datetime import date

import pytest

from knowledge.config import KNOWLEDGE_CHUNKING_VERSION, ROOT
from knowledge.models import KnowledgeSourceType as Kind
from knowledge.sources import KnowledgeCatalog, chunk_markdown
from tests.knowledge.index._support import source
from tests.knowledge.index._support import store as store


def test_chunking_stable_heading_paragraph_size_and_metadata():
    text = (
        "# Demo\n\nIntro.\n\n## Collision Damage\n\n"
        + "A vehicle hits an object. " * 20
        + "\n\n### Evidence\n\nPhotographs are needed."
    )
    first = chunk_markdown(source(), text, 100)
    assert first == chunk_markdown(source(), text, 100)
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(0 < len(c.text) <= 100 for c in first)
    assert all(c.chunking_version == KNOWLEDGE_CHUNKING_VERSION and c.product == "AUTO_PREMIUM" for c in first)
    assert first[-1].section == "Collision Damage / Evidence"
    assert all(c.source_version == "1" and c.effective_from == date(2026, 1, 1) for c in first)
    assert first != chunk_markdown(source(version="2"), text, 100)
    assert first != chunk_markdown(source(), text, 150)
    assert chunk_markdown(source(), "# Title\n\n ") == []
    long_token = chunk_markdown(source(), "## Token\n\n" + "a" * 250, 100)
    assert [len(c.text) for c in long_token] == [100, 100, 50]


def test_catalog_demo_and_validation(tmp_path):
    catalog = KnowledgeCatalog.load(ROOT / "data/knowledge")
    assert len(catalog.sources) == 3
    assert {s.source_type for s in catalog.sources} == {Kind.POLICY, Kind.PROCEDURE}
    assert len(catalog.chunks) > 10
    (tmp_path / "policies").mkdir()
    path = tmp_path / "policies/test.md"
    path.write_text("Not a synthetic source", encoding="utf-8")
    metadata = source().model_dump(mode="json", exclude={"file_path"})
    import json

    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic"):
        KnowledgeCatalog.load(tmp_path)
    with pytest.raises(ValueError, match="No demo"):
        KnowledgeCatalog.load(tmp_path / "absent")
