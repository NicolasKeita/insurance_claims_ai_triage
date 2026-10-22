"""Keep failed Qdrant checks in the report without duplicate constructor warnings."""

from evaluation.preflight import RealEvaluationConfig
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
