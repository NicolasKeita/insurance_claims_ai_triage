"""Portable demo review uses production documents and preserves missing inputs."""

import pytest

from claims.review.demo.preparation import prepare_demo
from claims.review.demo.provider import DemoReviewService
from persistence.repositories import ClaimNotFoundError


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("review-demo")
    prepare_demo(root)
    return root


def test_minimal_cases_preserve_partial_evidence(prepared):
    service = DemoReviewService(prepared)
    assert len(service.list_demo_claims()) == 3
    coherent = service.get_review("E2E-COHERENT-0001")
    assert coherent.consistency.data["claim_id_match"] is True
    assert coherent.investigation.data["signals"] == []
    assert coherent.investigation.data["partial"] is True
    assert coherent.anomaly.status == "NOT_EXECUTED" and coherent.anomaly.data is None
    assert coherent.history.status == "UNAVAILABLE" and coherent.history.data is None
    assert coherent.agent.status == coherent.vision.status == "NOT_EXECUTED"
    assert all(item["extraction_method"] == "NATIVE" for item in coherent.documents.data["items"])
    quote = service.get_review("E2E-QUOTE-0001")
    assert quote.consistency.data["additional_quote_damage"] == ["HOOD"]
    assert quote.investigation.data["signals"][0]["code"] == "ADDITIONAL_QUOTE_DAMAGE"
    missing = service.get_review("E2E-MISSING-0001")
    assert missing.consistency.status == "AVAILABLE"
    absent = [item for item in missing.documents.data["items"] if not item["available"]]
    assert len(absent) == 1 and absent[0]["type"] == "POLICE_REPORT"
    assert absent[0]["extraction_method"] is None and absent[0]["extracted_data"] is None
    assert any(item["code"] == "MISSING_DOCUMENTS" for item in missing.investigation.data["signals"])


def test_review_requests_do_not_recompute_or_mutate_snapshots(prepared, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Review GET must never materialize or extract a case")

    monkeypatch.setattr("claims.review.demo.preparation.materialize_scenario_case", unexpected)
    monkeypatch.setattr("claims.review.demo.preparation.extract_case_documents", unexpected)
    service = DemoReviewService(prepared)
    first = service.get_review("E2E-COHERENT-0001")
    first.documents.data["items"].clear()
    second = service.get_review("E2E-COHERENT-0001")
    assert len(second.documents.data["items"]) == 4
    assert first.generated_at == second.generated_at
    with pytest.raises(ClaimNotFoundError):
        service.get_review("UNKNOWN")


def test_setup_and_existing_outputs_fail_without_overwriting(prepared, tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_review_demo"):
        DemoReviewService(tmp_path / "missing")
    before = (prepared / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_demo(prepared)
    assert (prepared / "manifest.json").read_bytes() == before


def test_snapshot_identity_and_asset_containment_are_enforced(prepared):
    service = DemoReviewService(prepared)
    entry = service.entries["E2E-COHERENT-0001"]
    entry["snapshot"] = "additional_quote_damage.json"
    with pytest.raises(ValueError, match="identity"):
        service.get_review("E2E-COHERENT-0001")
    entry["snapshot"] = "../outside.json"
    with pytest.raises(ValueError, match="leaves"):
        service.get_review("E2E-COHERENT-0001")
    entry["images"] = {"escape.png": "../outside.png"}
    assert service.image_path("E2E-COHERENT-0001", "escape.png") is None
    assert service.image_path("E2E-COHERENT-0001", "unknown.png") is None
