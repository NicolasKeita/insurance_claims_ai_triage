"""Test cache."""

import pytest


def test_missing_rag_cache_is_checked_without_loading_or_downloading_a_model(monkeypatch):
    import huggingface_hub

    import tests.infrastructure.integration_prerequisites._cache as support

    calls = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda model, name: calls.append(name))
    with pytest.raises(support.PrerequisiteError, match="not present in the local cache"):
        support.require_cached_rag_model()
    assert calls == ["modules.json"]
