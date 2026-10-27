"""Malformed snapshots cannot become negative facts or leak private read errors."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from claims.investigation.history import HistoricalClaimProfile
from claims.review.assembly import build_review
from claims.review.demo.observation_fields import filter_agent_similar, filter_similar, validated_history
from claims.review.models import ReviewSection


@pytest.mark.parametrize("error", [OSError("private path"), ValueError("private snapshot"), KeyError("private key")])
@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/review/demo-claims",
        "/v1/claims/claim/review",
        "/v1/claims/claim/review/images/photo.png",
    ],
)
def test_snapshot_read_failures_are_sanitized_service_unavailability(error, endpoint):
    class Service:
        def fail(self, *args):
            raise error

        get_review = list_demo_claims = image_path = fail

    with TestClient(create_app(AppConfig(review_service_loader=lambda _: Service()))) as client:
        response = client.get(endpoint)
    assert response.status_code == 503
    assert response.json() == {"detail": "Stored claim review data is temporarily unavailable"}
    assert "private" not in response.text


@pytest.mark.parametrize("partial", [{}, {"previous_claim_count": 0}])
def test_incomplete_history_never_defaults_to_zero(partial):
    with pytest.raises(ValueError, match="incomplete"):
        validated_history(partial)
    complete = HistoricalClaimProfile().model_dump(mode="json")
    assert validated_history(complete) == complete
    assert validated_history(None) is None


def test_saved_comparison_and_audit_share_strict_history_filter(reference_claim):
    earlier = str(reference_claim.incident.date - timedelta(days=1))
    candidates = [
        {"claim_id": "earlier", "incident_date": earlier},
        {"claim_id": "same-day", "incident_date": str(reference_claim.incident.date)},
        {"claim_id": "future", "incident_date": "2099-01-01"},
        {"claim_id": reference_claim.claim_id, "incident_date": earlier},
    ]
    assert filter_similar({"results": candidates}, reference_claim)["results"] == candidates[:1]
    original = {"evidence": [{"source_type": "SIMILAR_CLAIM", "structured_data": item} for item in candidates]}
    assert len(filter_agent_similar(original, reference_claim)["evidence"]) == 1
    assert len(original["evidence"]) == 4


def test_bounded_agent_signal_json_never_exposes_local_paths_or_raw_summaries(reference_claim):
    facts = '{"asset_reference": "C:\\\\private\\\\claim.png", "additional_damage": ["HOOD"]}'
    evidence = {
        "evidence_id": "investigation:stored-record",
        "source_type": "INVESTIGATION_ASSESSMENT",
        "source_reference": "stored-record",
        "summary": f"Stored review; signals: {facts[:70]}",
        "structured_data": {"signal_details": [{"facts": facts, "facts_truncated": True}], "review_score": 22},
    }
    view = build_review(reference_claim, {"agent": ReviewSection(status="AVAILABLE", data={"evidence": [evidence]})})
    public = view.model_dump_json()
    assert "private" not in public and "signal_details" not in public
    item = next(item for item in view.evidence if item.id == evidence["evidence_id"])
    assert item.title == "Investigation assessment"
    assert item.data == {"review_score": 22}
    assert view.agent.data["evidence"][0]["summary"] == (
        "Investigation assessment. See the structured assessment evidence."
    )
    assert evidence["structured_data"]["signal_details"]
