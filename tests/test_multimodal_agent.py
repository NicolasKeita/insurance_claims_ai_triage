"""The agent retrieves existing visual evidence and cannot request inference."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_agent_tools import plan
from test_multimodal_investigation import make_visual_assessment

import agent.tools as adapters
import agent.tools.base as tool_base
import agent.tools.observations as tool_observations
from agent.tools import ReadOnlyAgentTools
from persistence.vision_assessments import StoredVisionAssessment


@pytest.mark.parametrize("available", [True, False])
def test_visual_tool_reads_existing_snapshot_only_and_closes_session(configured, monkeypatch, available):
    claim, factory, calls, _ = configured
    record = StoredVisionAssessment(uuid4(), datetime.now(timezone.utc), make_visual_assessment())

    class Vision:
        def __init__(self, session):
            assert session.open_count == 1
            self.session = session

        def latest_for_claim(self, claim_id):
            tool_base.ClaimRepository(self.session).require_pk(claim_id)
            calls.append(("vision", claim_id))
            return record if available else None

        def add(self, *_):
            raise AssertionError("The agent must never compute or persist visual assessments")

    monkeypatch.setattr(tool_observations, "VisionAssessmentRepository", Vision)
    result = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_VISUAL_ASSESSMENT"))
    assert factory.open_count == 0
    if not available:
        assert result.status == "NOT_AVAILABLE"
        assert result.evidence == []
        return
    assert result.evidence[0].evidence_id == f"vision:{record.id}"
    assert result.evidence[0].source_type == "VISUAL_ASSESSMENT"
    assert result.data["consistency"]["declared_damage_not_visually_confirmed"] == ["FRONT_BUMPER"]
    assert result.data["consistency"]["visually_supported_quote_damage"] == ["HOOD"]
    assert result.data["observations"][0]["image"] == "fixture-0.jpg"
    assert "asset_reference" not in result.model_dump_json()
    assert "Non-detection is not evidence of absence" in result.evidence[0].summary


@pytest.mark.parametrize("arguments", [{"claim_id": "other"}, {"checkpoint": "best.pt"}, {"run_inference": True}])
def test_visual_tool_cannot_change_claim_or_trigger_inference(arguments):
    with pytest.raises(ValidationError):
        plan("GET_VISUAL_ASSESSMENT", **arguments)
