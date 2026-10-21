"""Test transactions."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api.main import AppConfig, create_app
from persistence.models import (
    AnomalyAssessmentRow,
    ClaimRow,
    InvestigationAssessmentRow,
)
from persistence.repositories import AssessmentRepository, ClaimRepository
from persistence.service import PersistenceService
from tests.domain.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.domain.investigation.policy._support import make_consistency
from tests.domain.persistence.mapping._support import make_anomaly


def test_seed_rejects_unexpected_fixture_without_overwriting(factory, reference_claim):
    from scripts.seed_historical_claims import seed

    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        collision = reference_claim.model_copy(update={"claim_id": "DEMO-STEP25-HISTORY-P03"})
        ClaimRepository(session).add(collision)
        with pytest.raises(ValueError, match="Unexpected data"):
            seed(session)
        assert ClaimRepository(session).get_by_claim_id(collision.claim_id) == collision
        assert session.scalar(select(func.count()).select_from(ClaimRow)) == 2


def test_api_rolls_back_anomaly_if_investigation_write_fails(factory, reference_claim, monkeypatch):
    from sqlalchemy.exc import IntegrityError

    PersistenceService(factory).import_claim(reference_claim)

    class Predictor:
        def assess(self, features):
            return make_anomaly()

    def fail(*args, **kwargs):
        raise IntegrityError("demo failure", {}, Exception("write rejected"))

    monkeypatch.setattr(AssessmentRepository, "add_investigation", fail)
    app = create_app(AppConfig(anomaly_predictor_loader=lambda: Predictor(), session_factory=factory))
    with TestClient(app) as client:
        result = client.post(
            f"/v1/claims/{reference_claim.claim_id}/investigation/assess",
            json={"consistency": make_consistency().model_dump(mode="json")},
        )
        assert result.status_code == 503
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(AnomalyAssessmentRow)) == 0
        assert session.scalar(select(func.count()).select_from(InvestigationAssessmentRow)) == 0
