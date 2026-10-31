"""Source adapters preserve exact case identity and never materialize missing assets."""

import pytest

from claims.review.archive.sources import case_sources, prepared_sources, selected_cases
from claims.review.demo.provider import DemoReviewService
from tests.claims.review_archive._support import IMAGE, PDF, recorded_review
from tests.claims.review_archive.cli._support import write_case, write_prepared


def test_case_sources_requires_matching_recorded_facts_and_cannot_substitute_another_case(tmp_path):
    review = recorded_review()
    other = recorded_review()
    other.claim.repair_estimate.amount += 1
    root = write_case(tmp_path / "other-case", other)
    with pytest.raises(ValueError, match="facts differ"):
        case_sources(review, root)
    other.claim.claim_id = "ANOTHER-CLAIM"
    (root / "claim.json").write_text(other.claim.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="facts differ"):
        case_sources(review, root)


def test_selected_cases_rejects_ambiguous_duplicate_claim_sources(tmp_path):
    review = recorded_review()
    one = write_case(tmp_path / "one", review)
    two = write_case(tmp_path / "two", review)
    assert selected_cases([one]) == {"CLAIM-0001": one.absolute()}
    with pytest.raises(ValueError, match="one original"):
        selected_cases([one, two])


def test_case_sources_rejects_unsafe_claim_identity_before_building_paths(tmp_path):
    review = recorded_review()
    review.claim.claim_id = "../other-claim"
    with pytest.raises(ValueError, match="identifier"):
        case_sources(review, tmp_path / "original")


def test_source_claim_json_symlink_is_rejected(tmp_path):
    review = recorded_review()
    original = write_case(tmp_path / "original", review)
    selected = tmp_path / "linked-case"
    selected.mkdir()
    try:
        (selected / "claim.json").symlink_to(original / "claim.json")
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable in this environment: {error}")
    with pytest.raises(ValueError, match="symlink"):
        case_sources(review, selected)


def test_prepared_sources_uses_existing_fixture_bytes_without_rebuilding_them(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Export must not materialize source fixtures or run extraction")

    monkeypatch.setattr("evaluation.deterministic.fixtures.materialize_scenario_case", forbidden)
    review = recorded_review()
    root = write_prepared(tmp_path / "prepared", [review])
    service = DemoReviewService(root)
    documents, images = prepared_sources(service, review, {})
    assert documents["declaration finale.pdf"].read_bytes() == PDF
    assert images["photo originale.png"].read_bytes() == IMAGE
    assert "police.pdf" not in documents
    (root / "fixtures" / "source-case-0" / "documents" / "declaration finale.pdf").unlink()
    documents, _ = prepared_sources(service, review, {})
    assert not documents["declaration finale.pdf"].exists()


def test_explicit_case_cannot_override_prepared_review_with_different_business_facts(tmp_path):
    review = recorded_review()
    root = write_prepared(tmp_path / "prepared", [review])
    other = recorded_review()
    other.claim.vehicle.year = 2020
    selected = write_case(tmp_path / "different-original", other)
    with pytest.raises(ValueError, match="facts differ"):
        prepared_sources(DemoReviewService(root), review, {"CLAIM-0001": selected})
