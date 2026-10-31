"""Exports preserve recorded results and copy only explicit original assets."""

import hashlib
import json
from pathlib import Path

import pytest

from claims.review.archive import export_review
from tests.claims.review_archive._support import IMAGE, PDF, recorded_review, source_assets


def load_manifest(bundle):
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def test_copies_original_bytes_and_preserves_frozen_results_without_mutating_source(tmp_path):
    review = recorded_review()
    before = review.model_dump_json()
    documents, images = source_assets(tmp_path / "originals")
    bundle = export_review(
        review,
        output_root=tmp_path / "analysis_results",
        analysis_id="ANALYSIS-0003",
        title="Collision avant — Renault Clio",
        document_paths=documents,
        image_paths=images,
        source={"kind": "recorded-review", "reference": "test-snapshot"},
    )
    assert bundle.parent == tmp_path / "analysis_results"
    assert "CLAIM-0001" in bundle.name and "ANALYSIS-0003" in bundle.name
    manifest = load_manifest(bundle)
    assert manifest["claim_id"] == "CLAIM-0001"
    assert manifest["analysis_id"] == "ANALYSIS-0003"
    copied = [entry for entry in manifest["files"] if entry["status"] == "COPIED"]
    assert {entry["kind"] for entry in copied} == {"document", "image"}
    expected = {"declaration finale.pdf": PDF, "photo originale.png": IMAGE}
    for entry in copied:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        target = bundle / relative
        assert target.read_bytes() == expected[entry["filename"]]
        assert not target.is_symlink()
        original = (documents | images)[entry["filename"]]
        assert not target.samefile(original)
        assert entry["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert entry["size_bytes"] == target.stat().st_size
    results = json.loads((bundle / "resultats.json").read_text(encoding="utf-8"))
    assert results["investigation"]["source"] == "investigation:frozen-001"
    assert results["vision"]["source"] == "vision:frozen-002"
    assert results["agent"]["data"]["run_id"] == "stable-run-003"
    assert results["agent"]["data"]["recommendation"] == review.agent.data["recommendation"]
    assert results["anomaly"]["status"] == "NOT_EXECUTED"
    assert review.model_dump_json() == before
    assert (bundle / "rapport.html").is_file()
    assert (bundle.parent / "index.html").is_file()


def test_nonprovided_document_is_not_substituted_even_if_an_unrelated_source_is_supplied(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    documents["police.pdf"] = documents["declaration finale.pdf"]
    bundle = export_review(
        recorded_review(),
        output_root=tmp_path / "analysis_results",
        analysis_id="MISSING-POLICE-REPORT",
        document_paths=documents,
        image_paths=images,
    )
    police = next(entry for entry in load_manifest(bundle)["files"] if entry["filename"] == "police.pdf")
    assert police["status"] == "NOT_PROVIDED"
    assert not (bundle / "documents" / "police.pdf").exists()


def test_registered_source_that_does_not_exist_is_marked_missing(tmp_path):
    bundle = export_review(
        recorded_review(),
        output_root=tmp_path / "analysis_results",
        analysis_id="UNAVAILABLE-PDF",
        document_paths={"declaration finale.pdf": tmp_path / "does-not-exist.pdf"},
    )
    document = next(entry for entry in load_manifest(bundle)["files"] if entry["filename"] == "declaration finale.pdf")
    assert document["status"] == "MISSING"
    assert document["reason"]
    assert not (bundle / "documents" / "declaration finale.pdf").exists()


def test_missing_assets_are_explicit_without_inventing_files_or_an_analysis(tmp_path):
    review = recorded_review()
    bundle = export_review(review, output_root=tmp_path / "analysis_results", analysis_id="NO-ASSETS")
    manifest = load_manifest(bundle)
    assert manifest["files"]
    assert all(entry["status"] != "COPIED" for entry in manifest["files"])
    assert all(entry.get("reason") for entry in manifest["files"])
    assert not list(bundle.rglob("*.pdf"))
    assert not list(bundle.rglob("*.png"))
    results = json.loads((bundle / "resultats.json").read_text(encoding="utf-8"))
    assert results["investigation"]["data"] == review.investigation.data
    assert results["vision"]["data"]["images"][0]["image_url"] is None


def test_export_cannot_overwrite_an_existing_analysis(tmp_path):
    review = recorded_review()
    output = tmp_path / "analysis_results"
    first = export_review(review, output_root=output, analysis_id="FIXED-ANALYSIS")
    marker = first / "keep-user-note.txt"
    marker.write_text("Do not overwrite", encoding="utf-8")
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        export_review(review, output_root=output, analysis_id="FIXED-ANALYSIS")
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before == after
