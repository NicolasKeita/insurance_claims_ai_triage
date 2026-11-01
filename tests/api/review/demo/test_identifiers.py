"""Demo references stay readable without rewriting saved evidence or real IDs."""

import json
import re

import pytest

from claims.review.archive import export_review
from claims.review.archive.provider import ArchiveReviewService
from claims.review.demo.provider import DEMO_FORMAT, DemoReviewService
from evaluation.deterministic.document_generation.identifiers import display_identifiers, fake_identifiers
from evaluation.scenarios import load_scenarios
from tests.claims.review_archive._support import recorded_review


def legacy_review():
    review = recorded_review()
    review.mode = "demo"
    review.claim.customer.customer_id = "DEMO-E2E-CUSTOMER-visual_additional_damage"
    review.claim.policy.policy_id = "DEMO-E2E-POLICY-visual_additional_damage"
    return review


def test_synthetic_references_are_stable_numeric_and_distinct_across_scenarios():
    references = [fake_identifiers(item.scenario_id) for item in load_scenarios()]
    assert references == [fake_identifiers(item.scenario_id) for item in load_scenarios()]
    assert len({customer for customer, _ in references}) == len(references)
    assert len({policy for _, policy in references}) == len(references)
    assert all(re.fullmatch(r"CLI-\d{10}", customer) for customer, _ in references)
    assert all(re.fullmatch(r"AUTO-\d{10}", policy) for _, policy in references)


def test_legacy_aliases_match_new_fixtures_without_mutating_the_original_review():
    review = legacy_review()
    frozen = review.model_dump_json()
    displayed = display_identifiers(review)
    customer, policy = fake_identifiers("visual_additional_damage")
    assert displayed.claim.customer.customer_id == customer
    assert displayed.claim.policy.policy_id == policy
    assert review.model_dump_json() == frozen
    assert displayed.claim.model_dump(exclude={"customer", "policy"}) == review.claim.model_dump(
        exclude={"customer", "policy"}
    )
    assert displayed.model_dump(exclude={"claim"}) == review.model_dump(exclude={"claim"})


@pytest.mark.parametrize("mode", ["demo", "persisted"])
def test_unrelated_customer_and_policy_references_are_preserved(mode):
    review = recorded_review()
    review.mode = mode
    assert display_identifiers(review).model_dump() == review.model_dump()


def test_persisted_legacy_references_are_never_aliased():
    review = legacy_review()
    review.mode = "persisted"
    assert display_identifiers(review) is review


def test_demo_archive_aliases_preserve_stored_payload_checksums_and_assessments(tmp_path):
    review = legacy_review()
    bundle = export_review(review, output_root=tmp_path / "archives", analysis_id="FROZEN")
    snapshot = bundle / "resultats.json"
    frozen = snapshot.read_bytes()
    archived = ArchiveReviewService(bundle).get_review(review.claim.claim_id)
    displayed = DemoReviewService(bundle).get_review(review.claim.claim_id)
    customer, policy = fake_identifiers("visual_additional_damage")
    assert displayed.claim.customer.customer_id == customer
    assert displayed.claim.policy.policy_id == policy
    assert archived.claim.customer.customer_id.startswith("DEMO-E2E-CUSTOMER-")
    assert displayed.model_dump(exclude={"claim"}) == archived.model_dump(exclude={"claim"})
    assert snapshot.read_bytes() == frozen
    assert ArchiveReviewService(bundle).get_review(review.claim.claim_id).claim == archived.claim


def test_prepared_legacy_snapshot_is_aliased_without_rewriting_the_file(tmp_path):
    review = legacy_review()
    snapshot = tmp_path / "legacy.json"
    snapshot.write_text(review.model_dump_json(), encoding="utf-8")
    frozen = snapshot.read_bytes()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": DEMO_FORMAT,
                "claims": [{"claim_id": review.claim.claim_id, "snapshot": snapshot.name}],
            }
        ),
        encoding="utf-8",
    )
    displayed = DemoReviewService(tmp_path).get_review(review.claim.claim_id)
    customer, policy = fake_identifiers("visual_additional_damage")
    assert displayed.claim.customer.customer_id == customer
    assert displayed.claim.policy.policy_id == policy
    assert snapshot.read_bytes() == frozen
