"""Archive-backed HTTP reads preserve selected analyses and copied source bytes."""

from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from claims.review.archive import export_review
from claims.review.demo.provider import DemoReviewService
from tests.claims.review_archive._support import IMAGE, PDF, recorded_review, source_assets


def forbidden(*args, **kwargs):
    raise AssertionError("Reading an archive must not initialize inference or retrieval")


def config(service):
    return AppConfig(
        review_service_loader=lambda _: service,
        predictor_loader=forbidden,
        anomaly_predictor_loader=forbidden,
        similar_service_loader=forbidden,
        knowledge_retriever_loader=forbidden,
        knowledge_rag_loader=forbidden,
        agent_service_loader=forbidden,
    )


def archived_versions(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    review, output = recorded_review(), tmp_path / "analysis_results"
    first = export_review(
        review,
        output_root=output,
        analysis_id="FIRST",
        document_paths=documents,
        image_paths=images,
    )
    review.agent.data["run_id"] = "second-recorded-run"
    documents["declaration finale.pdf"].write_bytes(PDF + b"second PDF version\n")
    export_review(
        review,
        output_root=output,
        analysis_id="SECOND",
        document_paths=documents,
        image_paths=images,
    )
    return DemoReviewService(output), first


def test_selected_analysis_and_its_source_urls_remain_pinned_after_a_later_export(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    service, first = archived_versions(tmp_path)
    before = (first / "resultats.json").read_bytes()
    with TestClient(create_app(config(service))) as client:
        catalogue = client.get("/v1/review/demo-claims").json()
        assert [item["claim_id"] for item in catalogue] == ["CLAIM-0001"]
        response = client.get("/v1/claims/CLAIM-0001/review?analysis=FIRST")
        assert response.status_code == 200
        view = response.json()
        assert view["agent"]["data"]["run_id"] == "stable-run-003"
        document_url = view["documents"]["data"]["items"][0]["document_url"]
        assert "analysis=FIRST" in document_url
        pdf = client.get(document_url)
        assert pdf.status_code == 200 and pdf.content == PDF
        assert pdf.headers["content-type"] == "application/pdf"
        assert pdf.headers["content-disposition"].startswith("inline;")
        image_url = view["vision"]["data"]["images"][0]["image_url"]
        assert "analysis=FIRST" in image_url
        image = client.get(image_url)
        assert image.status_code == 200 and image.content == IMAGE
        latest = client.get("/v1/claims/CLAIM-0001/review").json()
        assert latest["agent"]["data"]["run_id"] == "second-recorded-run"
        assert client.get(latest["documents"]["data"]["items"][0]["document_url"]).content != PDF
        assert client.get("/v1/claims/CLAIM-0001/review?analysis=UNKNOWN").status_code == 404
    assert (first / "resultats.json").read_bytes() == before


def test_absent_or_modified_original_pdf_is_an_explicit_http_404(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    service, first = archived_versions(tmp_path)
    with TestClient(create_app(config(service))) as client:
        root = "/v1/claims/CLAIM-0001/review/documents/"
        assert client.get(root + "police.pdf?analysis=FIRST").status_code == 404
        assert client.get(root + "unregistered.pdf?analysis=FIRST").status_code == 404
        assert client.get(root + "active.html?analysis=FIRST").status_code == 404
        url = root + "declaration%20finale.pdf?analysis=FIRST"
        assert client.get(url).status_code == 200
        (first / "documents" / "declaration finale.pdf").write_bytes(b"tampered source")
        assert client.get(url).status_code == 404
