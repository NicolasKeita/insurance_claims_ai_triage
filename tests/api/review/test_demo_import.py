"""Saved-result import checks provenance, dates and absence without inference."""

import json
from functools import partial

import pytest

from claims.review.demo.observations import import_observations
from claims.review.demo.preparation import prepare_demo
from claims.review.demo.provider import DemoReviewService
from evaluation.deterministic.document_generation.identifiers import fake_identifiers
from persistence.artifact_storage import relocate_path


def saved_observation(tmp_path, observation, mode="real"):
    root = tmp_path / "saved-run"
    (root / "observations").mkdir(parents=True)
    path = root / "observations" / "simple_coherent_claim.json"
    path.write_text(json.dumps(observation), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps({"mode": mode, "created_at": "2026-09-25T11:00:00Z"}), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("mode,versions", [("deterministic", {}), ("real", {"triage": "fake-model-v1"})])
def test_import_rejects_deterministic_and_fake_observations(tmp_path, mode, versions):
    path = saved_observation(tmp_path, {"provenance": {"mode": mode}, "model_versions": versions}, mode)
    with pytest.raises(ValueError, match="real observations|fake model"):
        import_observations(tmp_path, [path])


def test_import_maps_saved_partial_status_and_keeps_unknown_metadata(tmp_path):
    minimal = tmp_path / "minimal"
    prepare_demo(minimal)
    claim = DemoReviewService(minimal).get_review("E2E-COHERENT-0001").claim
    observation = {
        "facts": {"claim": claim.model_dump(mode="json")},
        "provenance": {"mode": "real"},
        "components": {"anomaly": {"status": "FAIL"}, "history": {"status": "SKIP"}},
    }
    path = saved_observation(tmp_path, observation)
    imported = tmp_path / "imported"
    prepare_demo(imported, [path])
    review = DemoReviewService(imported).get_review(claim.claim_id)
    assert review.anomaly.status == "FAILED" and review.anomaly.data is None
    assert review.history.status == "UNAVAILABLE" and review.history.data is None
    assert review.triage.status == "NOT_EXECUTED" and review.triage.data is None
    assert review.documents.data["items"][0]["extraction_method"] is None
    assert review.audit.timestamps["documents"] == "2026-09-25T11:00:00+00:00"
    assert "planner_reason" not in review.model_dump_json()


def test_import_rejects_claim_identity_drift(tmp_path):
    minimal = tmp_path / "minimal"
    prepare_demo(minimal)
    claim = DemoReviewService(minimal).get_review("E2E-QUOTE-0001").claim
    path = saved_observation(
        tmp_path,
        {
            "facts": {"claim": claim.model_dump(mode="json")},
            "provenance": {"mode": "real"},
        },
    )
    with pytest.raises(ValueError, match="canonical scenario"):
        import_observations(tmp_path, [path])


def test_import_replaces_legacy_references_in_new_snapshots_only(tmp_path):
    minimal = tmp_path / "minimal"
    prepare_demo(minimal)
    claim = DemoReviewService(minimal).get_review("E2E-COHERENT-0001").claim
    claim.customer.customer_id = "DEMO-E2E-CUSTOMER-simple_coherent_claim"
    claim.policy.policy_id = "DEMO-E2E-POLICY-simple_coherent_claim"
    path = saved_observation(
        tmp_path,
        {"facts": {"claim": claim.model_dump(mode="json")}, "provenance": {"mode": "real"}},
    )
    frozen = path.read_bytes()
    imported = tmp_path / "imported"
    prepare_demo(imported, [path])
    payload = json.loads((imported / "saved-simple_coherent_claim.json").read_text(encoding="utf-8"))
    customer, policy = fake_identifiers("simple_coherent_claim")
    assert payload["claim"]["customer"]["customer_id"] == customer
    assert payload["claim"]["policy"]["policy_id"] == policy
    assert path.read_bytes() == frozen


def test_import_finds_relocated_saved_run_without_rewriting_observations(tmp_path, monkeypatch):
    from claims.review.demo import observations

    minimal = tmp_path / "minimal"
    prepare_demo(minimal)
    claim = DemoReviewService(minimal).get_review("E2E-COHERENT-0001").claim
    original = saved_observation(
        tmp_path / "artifacts/evaluation",
        {"facts": {"claim": claim.model_dump(mode="json")}, "provenance": {"mode": "real"}},
    )
    frozen = original.read_bytes()
    current = tmp_path / "artifacts/Evaluations"
    original.parents[2].rename(current)
    monkeypatch.setattr(observations, "relocate_path", partial(relocate_path, root=tmp_path))
    imported = tmp_path / "imported"

    prepare_demo(imported, [original])

    review = DemoReviewService(imported).get_review(claim.claim_id)
    assert review.claim.claim_id == claim.claim_id
    assert review.mode == "demo"
    assert (current / "saved-run/observations/simple_coherent_claim.json").read_bytes() == frozen
