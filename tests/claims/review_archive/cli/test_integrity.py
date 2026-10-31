"""An export preserves recorded image provenance and explicit analysis selection."""

import hashlib
import json

import pytest

from claims.review.archive import export_review
from claims.review.demo.provider import DEMO_FORMAT, DemoReviewService
from persistence.repositories import ClaimNotFoundError
from tests.claims.review_archive._support import IMAGE, recorded_review, source_assets


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_review", ("CLAIM-0001",)),
        ("image_path", ("CLAIM-0001", "photo originale.png")),
        ("document_path", ("CLAIM-0001", "declaration finale.pdf")),
    ],
)
def test_legacy_demo_never_silently_ignores_a_requested_analysis_identity(tmp_path, method, args):
    root = tmp_path / "legacy-demo"
    root.mkdir()
    (root / "review.json").write_text(recorded_review().model_dump_json(), encoding="utf-8")
    entry = {
        "claim_id": "CLAIM-0001",
        "title": "A legacy case",
        "description": "No analysis identity",
        "snapshot": "review.json",
        "images": {},
    }
    (root / "manifest.json").write_text(json.dumps({"format": DEMO_FORMAT, "claims": [entry]}), encoding="utf-8")
    service = DemoReviewService(root)
    assert service.get_review("CLAIM-0001").agent.data["run_id"] == "stable-run-003"
    with pytest.raises(ClaimNotFoundError):
        getattr(service, method)(*args, analysis_id="PINNED-ANALYSIS")


def test_original_image_hash_must_match_recorded_provenance_before_export_publication(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    review = recorded_review()
    review.vision.data["images"][0]["provenance"] = {"sha256": hashlib.sha256(IMAGE).hexdigest()}
    output = tmp_path / "analysis_results"
    bundle = export_review(
        review,
        output_root=output,
        analysis_id="VERIFIED",
        document_paths=documents,
        image_paths=images,
    )
    assert (bundle / "images" / "photo originale.png").read_bytes() == IMAGE
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    images["photo originale.png"].write_bytes(IMAGE + b"different original")
    with pytest.raises(ValueError):
        export_review(
            review,
            output_root=output,
            analysis_id="MISMATCHED",
            document_paths=documents,
            image_paths=images,
        )
    after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert before == after
    assert not (output / "CLAIM-0001_MISMATCHED").exists()
