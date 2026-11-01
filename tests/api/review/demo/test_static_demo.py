"""Standalone demo assets preserve every recorded result and source business fact."""

from pathlib import Path
from urllib.parse import unquote

from claims.assets import DocumentAsset
from claims.document_extraction import extract_claim_form
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.review.demo.provider import DemoReviewService
from evaluation.deterministic.document_generation.identifiers import fake_identifiers
from evaluation.reporting.static_demo import export_static_demo
from tests.api.review.demo._static_demo import make_archive, read_json, tree_bytes, without_urls


def test_export_includes_all_analyses_and_professional_pdfs_without_changing_recorded_results(tmp_path):
    first = make_archive(tmp_path)
    make_archive(tmp_path, analysis_id="SECOND")
    source = tmp_path / "archives"
    frozen = tree_bytes(source)
    output = export_static_demo(source, tmp_path / "demo")
    manifest = read_json(output / "manifest.json")
    assert manifest["format"] == "claims_static_demo_v1"
    assert "not recomputed" in manifest["scope"]
    assert [entry["analysis_id"] for entry in manifest["claims"]] == ["FIRST", "SECOND"]
    original = read_json(first / "manifest.json")
    assert manifest["claims"][0]["source_fingerprint"] == original["fingerprint"]
    assert manifest["claims"][0]["source_review_sha256"] == original["review_sha256"]
    files = manifest["claims"][0]["files"]
    document_record = next(item for item in files if item["kind"] == "document" and item["status"] == "COPIED")
    image_record = next(item for item in files if item["kind"] == "image")
    assert document_record["original_sha256"] != document_record["presentation_sha256"]
    assert image_record["original_sha256"] == image_record["presentation_sha256"]
    catalogue = read_json(output / "catalog.json")
    assert len(catalogue) == 1 and "SECOND" in catalogue[0]["description"]
    view = read_json(output / manifest["claims"][0]["review"])
    recorded = DemoReviewService(source).get_review("CLAIM-0001", "FIRST")
    assert without_urls(view) == without_urls(recorded.model_dump(mode="json"))
    customer, policy = fake_identifiers("visual_additional_damage")
    assert view["claim"]["customer"]["customer_id"] == customer
    assert view["claim"]["policy"]["policy_id"] == policy
    document_url = view["documents"]["data"]["items"][0]["document_url"]
    assert document_url.startswith("demo/documents/CLAIM-0001_FIRST/")
    document = output.parent / unquote(document_url)
    before = DocumentAsset(DocumentType.CLAIM_FORM, document.name, first / "documents" / document.name)
    after = DocumentAsset(DocumentType.CLAIM_FORM, document.name, document)
    assert extract_claim_form(extract_pdf_text(before)) == extract_claim_form(extract_pdf_text(after))
    assert "ALIZE ASSURANCES" in extract_pdf_text(after).text
    assert not (document.parent / "police.pdf").exists()
    image_url = view["vision"]["data"]["images"][0]["image_url"]
    image = output.parent / unquote(image_url)
    assert image.read_bytes() == (first / "images" / image.name).read_bytes()
    assert tree_bytes(source) == frozen


def test_export_is_deterministic_and_reuses_unchanged_outputs(tmp_path):
    make_archive(tmp_path)
    output = export_static_demo(tmp_path / "archives", tmp_path / "demo")
    frozen = tree_bytes(output)
    modified = (output / "manifest.json").stat().st_mtime_ns
    assert export_static_demo(tmp_path / "archives", output) == output
    assert tree_bytes(output) == frozen
    assert (output / "manifest.json").stat().st_mtime_ns == modified


def test_existing_professional_documents_are_copied_without_duplicate_stationery(tmp_path):
    from evaluation.reporting.static_demo.documents import restyle_document

    make_archive(tmp_path)
    output = export_static_demo(tmp_path / "archives", tmp_path / "demo")
    document = next((output / "documents").rglob("*.pdf"))
    target = tmp_path / Path(document.name)
    restyle_document(document, target, "CLAIM_FORM")
    assert target.read_bytes() == document.read_bytes()
