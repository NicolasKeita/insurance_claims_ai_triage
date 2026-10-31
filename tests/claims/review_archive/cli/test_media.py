"""Persisted claim documents expose only registered originals under their root."""

import pytest

from claims.enums import DocumentType
from claims.models import ClaimDocument
from persistence.repositories import ClaimNotFoundError
from tests.api.review._support import configure_repositories
from tests.claims.review_archive._support import PDF


def test_persisted_pdf_urls_keep_registered_inventory_and_bound_source_files(tmp_path, monkeypatch, reference_claim):
    claim = reference_claim.model_copy(
        update={
            "documents": [
                ClaimDocument(type=DocumentType.CLAIM_FORM, filename="declaration finale.pdf", available=True),
                ClaimDocument(type=DocumentType.POLICE_REPORT, filename="not-provided.pdf", available=False),
                ClaimDocument(type=DocumentType.REPAIR_QUOTE, filename="../private.pdf", available=True),
            ]
        }
    )
    before = claim.model_dump_json()
    root = tmp_path / "assets"
    documents = root / claim.claim_id / "documents"
    documents.mkdir(parents=True)
    pdf = documents / "declaration finale.pdf"
    pdf.write_bytes(PDF)
    service, _, _ = configure_repositories(monkeypatch, claim, asset_root=root)
    assert service.document_path(claim.claim_id, "declaration finale.pdf") == pdf.resolve()
    for filename in ("not-provided.pdf", "../private.pdf", "unregistered.pdf", "active.html"):
        assert service.document_path(claim.claim_id, filename) is None
    result = service.get_review(claim.claim_id).documents.data["items"]
    assert result[0]["document_url"].endswith("/review/documents/declaration%20finale.pdf")
    assert result[0]["document_unavailable_reason"] is None
    assert result[1]["document_url"] is None and result[1]["available"] is False
    assert result[2]["document_url"] is None
    assert result[2]["document_unavailable_reason"]
    assert claim.model_dump_json() == before


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_review", ()),
        ("image_path", ("photo.png",)),
        ("document_path", ("claim.pdf",)),
    ],
)
def test_persisted_latest_never_substitutes_a_requested_archived_analysis(monkeypatch, reference_claim, method, args):
    service, _, _ = configure_repositories(monkeypatch, reference_claim)
    with pytest.raises(ClaimNotFoundError):
        getattr(service, method)(reference_claim.claim_id, *args, analysis_id="ARCHIVED")
