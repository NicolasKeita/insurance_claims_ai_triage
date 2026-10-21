"""Test api."""

from fastapi.testclient import TestClient

from investigation.models import SignalCategory
from persistence.repositories import AssessmentRepository
from persistence.service import PersistenceService
from tests.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.investigation.history_postgres._support import (
    assert_history_requests,
    history_app,
)
from tests.persistence.mapping._support import make_anomaly


def test_api_history_and_persisted_v2_signals_snapshot(factory, reference_claim):
    """
    33 history + 12 missing documents

    Re-reading the stored snapshot preserves the old evidence exactly.
    """
    from scripts.seed_historical_claims import seed

    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    with factory.begin() as session:
        assert len(seed(session)) == 9
        assert all("Existing fixture verified" in message for message in seed(session))

    app, calls = history_app(factory)
    path = f"/v1/claims/{reference_claim.claim_id}"
    with TestClient(app) as client:
        body = assert_history_requests(client, path, reference_claim, calls)
    history = service.get_history(reference_claim.claim_id)
    assert len(history.investigation) == len(history.anomaly) == 2
    stored = history.investigation[0].result
    assert stored.model_dump(mode="json") == body
    historical_signals = [s for s in stored.signals if s.category == SignalCategory.HISTORICAL_CONTEXT]
    assert historical_signals[1].evidence["previous_anomalous_claim_count"] == 2
    with factory.begin() as session:
        result = make_anomaly()
        result.assessment.is_anomalous = False
        AssessmentRepository(session).add_anomaly("DEMO-STEP25-HISTORY-P03", result)
    assert service.get_historical_profile(reference_claim.claim_id).previous_anomalous_claim_count == 1
    assert service.get_history(reference_claim.claim_id).investigation[0].result == stored
