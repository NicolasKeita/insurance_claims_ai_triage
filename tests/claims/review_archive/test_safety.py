"""Archive boundaries reject unsafe paths and escape untrusted recorded text."""

import pytest

from claims.review.archive import export_review
from tests.claims.review_archive._support import recorded_review, source_assets


@pytest.mark.parametrize("unsafe", ["../escape", "C:\\private", "bad/name", "bad\\name", ".", "..", "CON", "trailing "])
def test_unsafe_analysis_identifier_never_writes_outside_output(tmp_path, unsafe):
    with pytest.raises(ValueError):
        export_review(recorded_review(), output_root=tmp_path / "analysis_results", analysis_id=unsafe)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("unsafe", ["../claim", "C:\\private", "CLAIM/0001", "CLAIM\\0001"])
def test_unsafe_claim_identifier_is_rejected(tmp_path, unsafe):
    review = recorded_review()
    review.claim.claim_id = unsafe
    with pytest.raises(ValueError):
        export_review(review, output_root=tmp_path / "analysis_results", analysis_id="SAFE-ID")


def test_supplied_html_cannot_be_disguised_as_an_exported_pdf(tmp_path):
    payload = tmp_path / "active.html"
    payload.write_text('<script>alert("untrusted")</script>', encoding="utf-8")
    with pytest.raises(ValueError):
        export_review(
            recorded_review(),
            output_root=tmp_path / "analysis_results",
            analysis_id="WRONG-KIND",
            document_paths={"declaration finale.pdf": payload},
        )


@pytest.mark.parametrize("unsafe", ["../escape.pdf", "C:\\private.pdf", "sub/file.pdf", "sub\\file.pdf"])
def test_asset_filename_traversal_is_rejected(tmp_path, unsafe):
    review = recorded_review()
    review.claim.documents[0].filename = unsafe
    with pytest.raises(ValueError):
        export_review(review, output_root=tmp_path / "analysis_results", analysis_id="SAFE-ID")
    assert not (tmp_path / "escape.pdf").exists()


def test_linked_source_asset_is_rejected_instead_of_followed(tmp_path):
    documents, images = source_assets(tmp_path / "originals")
    link = tmp_path / "linked.pdf"
    try:
        link.symlink_to(documents["declaration finale.pdf"])
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable in this environment: {error}")
    with pytest.raises(ValueError):
        export_review(
            recorded_review(),
            output_root=tmp_path / "analysis_results",
            analysis_id="LINKED-SOURCE",
            document_paths={"declaration finale.pdf": link},
            image_paths=images,
        )


def test_untrusted_title_and_recorded_text_are_escaped_in_offline_report(tmp_path):
    review = recorded_review()
    payload = '<script src="https://example.invalid/steal.js">alert(1)</script>'
    review.investigation.data["signals"][0]["explanation"] = payload
    review.agent.data["recommendation"]["case_summary"] = payload
    bundle = export_review(
        review,
        output_root=tmp_path / "analysis_results",
        analysis_id="ESCAPED",
        title=payload,
    )
    html = (bundle / "rapport.html").read_text(encoding="utf-8")
    assert payload not in html
    assert "&lt;script" in html
    assert 'src="https://example.invalid/' not in html
    assert "fetch(" not in html
