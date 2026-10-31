"""Read autonomous archives after moving them, preserving exact stored results."""

import json
import shutil

import pytest

from claims.review.archive import export_review
from claims.review.archive.provider import ArchiveReviewService
from tests.claims.review_archive._support import IMAGE, PDF, recorded_review, source_assets


def test_bundle_provider_reads_moved_copies_and_keeps_exact_assessment_references(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    bundle = export_review(
        recorded_review(),
        output_root=tmp_path / "analysis_results",
        analysis_id="FROZEN",
        document_paths=documents,
        image_paths=images,
    )
    relocated = tmp_path / "relocated" / bundle.name
    relocated.parent.mkdir()
    shutil.copytree(bundle, relocated)
    shutil.rmtree(tmp_path / "originals")
    shutil.rmtree(bundle)
    service = ArchiveReviewService(relocated)
    view = service.get_review("CLAIM-0001")
    assert view.investigation.source == "investigation:frozen-001"
    assert view.vision.source == "vision:frozen-002"
    assert view.agent.data["run_id"] == "stable-run-003"
    assert service.image_path("CLAIM-0001", "photo originale.png").read_bytes() == IMAGE
    assert service.document_path("CLAIM-0001", "declaration finale.pdf").read_bytes() == PDF
    document = next(item for item in view.documents.data["items"] if item["filename"] == "declaration finale.pdf")
    assert document["document_url"].startswith("/v1/claims/CLAIM-0001/review/documents/")
    assert view.vision.data["images"][0]["image_url"].startswith("/v1/claims/CLAIM-0001/review/images/")
    assert service.document_path("CLAIM-0001", "police.pdf") is None
    assert service.document_path("CLAIM-0001", "unregistered.pdf") is None
    assert service.image_path("CLAIM-0001", "unknown.png") is None


def test_collection_exposes_last_export_per_claim_without_changing_earlier_archive(tmp_path):
    review = recorded_review()
    output = tmp_path / "analysis_results"
    first = export_review(review, output_root=output, analysis_id="FIRST")
    frozen = (first / "resultats.json").read_bytes()
    review.agent.data["run_id"] = "later-agent-run"
    export_review(review, output_root=output, analysis_id="SECOND")
    service = ArchiveReviewService(output)
    assert [entry.claim_id for entry in service.list_demo_claims()] == ["CLAIM-0001"]
    assert service.get_review("CLAIM-0001").agent.data["run_id"] == "later-agent-run"
    assert (first / "resultats.json").read_bytes() == frozen
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "claim_analysis_collection_v1"
    assert len(manifest["analyses"]) == 2


def test_provider_cannot_serve_unregistered_or_traversing_assets(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    bundle = export_review(
        recorded_review(),
        output_root=tmp_path / "analysis_results",
        analysis_id="CONTAINED",
        document_paths=documents,
        image_paths=images,
    )
    service = ArchiveReviewService(bundle)
    assert service.document_path("CLAIM-0001", "../private.pdf") is None
    assert service.document_path("CLAIM-0001", "C:\\private.pdf") is None
    assert service.image_path("CLAIM-0001", "../private.png") is None


def test_modified_copied_source_is_not_served_as_the_recorded_original(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    bundle = export_review(
        recorded_review(),
        output_root=tmp_path / "analysis_results",
        analysis_id="CHECKSUM",
        document_paths=documents,
        image_paths=images,
    )
    service = ArchiveReviewService(bundle)
    service.document_path("CLAIM-0001", "declaration finale.pdf").write_bytes(PDF + b"modified")
    assert service.document_path("CLAIM-0001", "declaration finale.pdf") is None
    view = service.get_review("CLAIM-0001")
    assert view.documents.data["items"][0]["document_url"] is None
    assert service.image_path("CLAIM-0001", "photo originale.png").read_bytes() == IMAGE


def test_modified_snapshot_cannot_be_presented_as_the_frozen_analysis(tmp_path):
    bundle = export_review(recorded_review(), output_root=tmp_path / "analysis_results", analysis_id="FROZEN")
    service = ArchiveReviewService(bundle)
    snapshot = bundle / "resultats.json"
    snapshot.write_bytes(snapshot.read_bytes().replace(b"stable-run-003", b"changed-run-003"))
    with pytest.raises(ValueError, match="checksum"):
        service.get_review("CLAIM-0001")
