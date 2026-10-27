"""Partial failures, chronological context and original photograph allowlisting."""

import json

from sqlalchemy.exc import SQLAlchemyError

from claims.review.assembly import build_review
from claims.review.models import ReviewSection
from tests.api.review._support import configure_repositories


def test_component_failure_does_not_turn_into_a_negative_result(monkeypatch, reference_claim):
    service, records, _ = configure_repositories(monkeypatch, reference_claim, complete=True)
    records["anomaly"] = SQLAlchemyError("private database details")
    records["agent"] = ValueError("unsafe persisted run")
    view = service.get_review(reference_claim.claim_id)
    assert view.anomaly.status == "UNAVAILABLE"
    assert view.anomaly.data is None
    assert view.agent.status == "FAILED"
    assert view.triage.status == "AVAILABLE"
    assert "private" not in view.model_dump_json()


def test_stored_failed_agent_keeps_failure_distinct(monkeypatch, reference_claim):
    from tests.agent.persistence._support import make_agent_run

    service, records, _ = configure_repositories(monkeypatch, reference_claim)
    records["agent"] = make_agent_run(reference_claim.claim_id, failed=True)
    view = service.get_review(reference_claim.claim_id)
    assert view.agent.status == "FAILED"
    assert view.agent.data["recommendation"] is None
    assert view.agent.data["error"] == "The stored agent run failed."


def test_same_day_future_and_current_similar_claims_are_excluded(monkeypatch, reference_claim):
    from agent.models import EvidenceItem

    service, records, _ = configure_repositories(monkeypatch, reference_claim, complete=True)
    for identifier, when in [
        ("earlier", "2020-01-01"),
        ("same-day", str(reference_claim.incident.date)),
        ("future", "2099-01-01"),
    ]:
        records["agent"].evidence.append(
            EvidenceItem(
                evidence_id=f"similar:{identifier}",
                source_type="SIMILAR_CLAIM",
                source_reference=identifier,
                summary="Comparative context",
                structured_data={"claim_id": identifier, "incident_date": when},
            )
        )
    view = service.get_review(reference_claim.claim_id)
    assert [item["claim_id"] for item in view.similar_claims.data["results"]] == ["earlier"]
    assert {item.id for item in view.evidence}.isdisjoint({"similar:same-day", "similar:future"})
    assert "2099-01-01" not in view.model_dump_json()


def test_only_registered_images_under_claim_root_are_served(monkeypatch, reference_claim, tmp_path):
    claim = reference_claim.model_copy(update={"images": ["photo.png", "../outside.png"]})
    root = tmp_path / "assets"
    directory = root / claim.claim_id / "images"
    directory.mkdir(parents=True)
    photo = directory / "photo.png"
    photo.write_bytes(b"controlled image asset")
    service, _, _ = configure_repositories(monkeypatch, claim, asset_root=root)
    assert service.image_path(claim.claim_id, "photo.png") == photo.resolve()
    assert service.image_path(claim.claim_id, "../outside.png") is None
    assert service.image_path(claim.claim_id, "unregistered.png") is None
    image = service.get_review(claim.claim_id).vision.data["images"][0]
    assert image["image_url"].endswith("/review/images/photo.png")
    assert image["image_unavailable_reason"] is None


def test_explicit_states_and_sanitized_technical_provenance(reference_claim):
    view = build_review(
        reference_claim,
        {
            "knowledge": ReviewSection(
                status="INSUFFICIENT_EVIDENCE", data={"sources": [], "insufficient_evidence": True}
            ),
            "vision": ReviewSection(
                status="NO_EVIDENCE", data={"images": [], "asset_reference": "C:\\private\\photo.png"}
            ),
            "agent": ReviewSection(status="NOT_EXECUTED", data={"trace": [{"planner_reason": "private"}]}),
        },
    )
    assert view.knowledge.status == "INSUFFICIENT_EVIDENCE"
    assert view.vision.status == "NO_EVIDENCE"
    payload = json.loads(view.model_dump_json())
    assert "asset_reference" not in payload["vision"]["data"]
    assert "trace" not in payload["agent"]["data"]
