"""Support."""

from datetime import timedelta

from api.main import AppConfig, create_app
from persistence.repositories import ClaimRepository
from tests.investigation.policy._support import make_consistency
from tests.persistence.mapping._support import make_anomaly


def add_claim(session, reference, suffix, days, repair=None):
    payload = reference.model_dump()
    payload["claim_id"] = f"HISTORY-TEST-{suffix}"
    payload["incident"]["date"] = reference.incident.date - timedelta(days=days)
    payload["repair_estimate"] = (
        repair or reference.repair_estimate.model_validate({"amount": "100", "currency": "EUR"})
    ).model_dump()
    claim = type(reference).model_validate(payload)
    ClaimRepository(session).add(claim)
    return claim.claim_id


def history_app(factory):
    calls = []

    class Predictor:
        def assess(self, features):
            calls.append(features)
            result = make_anomaly()
            result.assessment.is_anomalous = False
            result.assessment.reference_percentile = 0.5
            return result

    def forbidden_triage():
        raise AssertionError("History/investigation does not need a triage model")

    app = create_app(
        AppConfig(
            predictor_loader=forbidden_triage, anomaly_predictor_loader=lambda: Predictor(), session_factory=factory
        )
    )
    return app, calls


def assert_history_requests(client, path, reference_claim, calls):
    response = client.get(path + "/history")
    assert response.status_code == 200
    assert response.json()["previous_claim_count"] == 6
    assert response.json()["previous_anomalous_claim_count"] == 2
    assert calls == []
    assert client.get("/v1/claims/unknown/history").status_code == 404
    assert (
        client.post(
            "/v1/claims/unknown/investigation/assess",
            json={"consistency": make_consistency().model_dump(mode="json")},
        ).status_code
        == 404
    )
    payload = {"consistency": make_consistency().model_dump(mode="json")}
    assert client.post(path + "/investigation/assess", json={**payload, "previous_claim_count": 999}).status_code == 422
    response = client.post(path + "/investigation/assess", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["policy_version"] == "investigation_policy_v2"
    assert body["review_score"] == 45
    assert body["review_priority"] == "ELEVATED"
    assert len(calls) == 1
    assert calls[0].repair_amount == float(reference_claim.repair_estimate.amount)
    assert client.post(path + "/investigation/assess", json=payload).json() == body
    return body
