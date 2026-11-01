"""Archive serving is read-only and explicitly ready only with existing analyses."""

import json

import pytest
from fastapi.testclient import TestClient

from api.review_app import create_app
from api.version import application_version
from claims.review.archive import export_review
from tests.claims.review_archive._support import IMAGE, PDF, recorded_review, source_assets


def test_readiness_and_registered_assets_work_without_database_or_mutating_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://invalid:unused@unavailable/unused")
    documents, images = source_assets(tmp_path / "originals")
    root = tmp_path / "analysis_results"
    export_review(recorded_review(), output_root=root, document_paths=documents, image_paths=images)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with TestClient(create_app(root)) as client:
        assert client.get("/readyz").json() == {"status": "ready", "mode": "archive", "claims": 1}
        assert client.get("/openapi.json").json()["info"]["version"] == application_version()
        view = client.get("/v1/claims/CLAIM-0001/review").json()
        assert view["agent"]["data"]["run_id"] == "stable-run-003"
        assert client.get(view["documents"]["data"]["items"][0]["document_url"]).content == PDF
        assert client.get(view["vision"]["data"]["images"][0]["image_url"]).content == IMAGE
        assert client.get("/v1/claims/unknown/review").status_code == 404
        assert client.post("/v1/claims/CLAIM-0001/review").status_code == 405
        assert client.post("/v1/claims/CLAIM-0001/agent/review").status_code == 404
        assert client.get("/health").status_code == 404
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before == after


def test_missing_archive_fails_startup_with_clear_instruction_and_does_not_generate_it(tmp_path):
    root = tmp_path / "missing-archives"
    with pytest.raises(RuntimeError, match="mount an existing, non-empty analysis_results"):
        with TestClient(create_app(root)):
            pass
    assert not root.exists()


def test_empty_archive_catalogue_fails_startup(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format": "claim_analysis_collection_v1", "analyses": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="valid manifest.json"):
        with TestClient(create_app(tmp_path)):
            pass


def test_corrupted_review_fails_readiness_before_serving_claims(tmp_path):
    root = tmp_path / "analysis_results"
    bundle = export_review(recorded_review(), output_root=root)
    (bundle / "resultats.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No analysis was generated"):
        with TestClient(create_app(root)):
            pass
