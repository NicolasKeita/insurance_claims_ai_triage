"""Keep failed Qdrant checks in the report without duplicate constructor warnings."""

from types import SimpleNamespace

import pytest

from evaluation.preflight import RealEvaluationConfig
from evaluation.preflight.console import print_preflight_row
from evaluation.preflight.session import PreflightSession


def test_failed_qdrant_probe_records_the_error_without_duplicate_compatibility_probe(monkeypatch):
    import qdrant_client

    options, calls = {}, []

    class OfflineClient:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def get_collections(self):
            calls.append("collections")
            raise TimeoutError("timed out")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(qdrant_client, "QdrantClient", OfflineClient)
    session = PreflightSession(RealEvaluationConfig(qdrant_url="http://127.0.0.1:6335"))
    session.probe_qdrant()
    assert options["check_compatibility"] is False
    assert calls == ["collections", "close"]
    assert not session.report.available("qdrant")
    assert "timed out" in session.report.reason("qdrant")


@pytest.mark.parametrize("installed_name", ["claims-e2e-partiel", "claims-e2e-partiel:latest"])
@pytest.mark.parametrize("details", [None, {"family": "gemma3"}, {"family": "gemma3", "parameter_size": "4.3B"}])
def test_ollama_alias_shows_selected_model_family(monkeypatch, capsys, installed_name, details):
    import httpx

    def tags(url, **kwargs):
        assert url == "http://ollama.test/api/tags"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"models": [
            {"name": "other", "details": {"family": "llama"}},
            {"name": installed_name, "details": details},
        ]})

    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test")
    monkeypatch.setenv("CLAIMS_LLM_MODEL", "other")
    monkeypatch.setattr(httpx, "get", tags)
    session = PreflightSession(
        RealEvaluationConfig(llm_model="claims-e2e-partiel"), progress=print_preflight_row,
    )
    session.probe_ollama()

    identity = ""
    if details:
        identity = "; gemma3" + (" 4.3B" if "parameter_size" in details else "")
    assert capsys.readouterr().out == f"  [OK] Ollama (claims-e2e-partiel{identity})\n"
    assert session.report.resources["llm_model"] == "claims-e2e-partiel"
    assert session.report.rows[0]["status"] == "PASS"
