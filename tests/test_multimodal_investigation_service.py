"""SQL orchestration receives completed perception; no checkpoint or inference."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from claims.loader import load_claim_case
import investigation.service as service
from investigation.history import HistoricalClaimProfile
from test_investigation import make_consistency
from test_multimodal_investigation import make_visual_assessment
from test_persistence import make_anomaly


def configure(monkeypatch):
    claim = load_claim_case(Path("data/CLAIM-2026-00001")).claim
    calls = []

    class Claims:
        def __init__(self, session):
            pass

        def get_by_claim_id(self, claim_id):
            calls.append("claim")
            return claim

    class History:
        def __init__(self, session):
            pass

        def get_profile(self, claim_id):
            calls.append("history")
            return HistoricalClaimProfile()

    class Vision:
        def __init__(self, session):
            pass

        def add(self, visual):
            calls.append("append_vision")

    class Assessments:
        def __init__(self, session):
            pass

        def add_anomaly(self, claim_id, result):
            calls.append("append_anomaly")

        def add_investigation(self, claim_id, result):
            calls.append("append_investigation")
            return SimpleNamespace(result=result)

    for name, value in (("ClaimRepository", Claims), ("HistoricalClaimRepository", History),
                        ("VisionAssessmentRepository", Vision), ("AssessmentRepository", Assessments)):
        monkeypatch.setattr(service, name, value)

    def anomaly(features):
        calls.append("anomaly")
        return make_anomaly()

    return claim, calls, anomaly


def test_multimodal_service_appends_completed_vision_and_v3_in_same_unit_of_work(monkeypatch):
    claim, calls, anomaly = configure(monkeypatch)
    visual = make_visual_assessment(declared=claim.declared_damage)
    result = service.assess_stored_claim_multimodal(
        object(), claim.claim_id, make_consistency(), anomaly, visual,
    )
    assert result.policy_version == "investigation_policy_v3"
    assert calls == ["claim", "history", "anomaly", "append_vision", "append_anomaly", "append_investigation"]
    assert "VISUAL_ADDITIONAL_DAMAGE" in {item.code for item in result.signals}


@pytest.mark.parametrize("invalid", ["foreign_claim", "stale_declaration"])
def test_multimodal_service_rejects_foreign_or_stale_snapshot_before_model_or_write(monkeypatch, invalid):
    claim, calls, anomaly = configure(monkeypatch)
    visual = make_visual_assessment(
        claim_id="other" if invalid == "foreign_claim" else claim.claim_id,
        declared=claim.declared_damage if invalid == "foreign_claim" else (),
    )
    with pytest.raises(ValueError):
        service.assess_stored_claim_multimodal(object(), claim.claim_id, make_consistency(), anomaly, visual)
    assert calls == ["claim"]
