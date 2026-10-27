"""Unavailable saved components cannot display stale negative assessments."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from claims.review.demo.state import knowledge_state, section
from claims.review.demo.vision import import_vision
from claims.review.models import ReviewSection


@pytest.mark.parametrize("execution,status", [("FAIL", "FAILED"), ("SKIP", "UNAVAILABLE")])
def test_saved_execution_state_removes_stale_negative_anomaly(execution, status):
    observation = {"components": {"anomaly": {"status": execution}}}
    assessment = section("anomaly", {"is_anomalous": False}, observation, "claim", datetime.now(timezone.utc))
    assert assessment.status == status
    assert assessment.data is None


@pytest.mark.parametrize("execution,status", [("FAIL", "FAILED"), ("SKIP", "UNAVAILABLE")])
def test_failed_vision_retains_no_stale_observations(tmp_path, execution, status):
    sections = {}
    observation = {
        "components": {"vision": {"status": execution}},
        "facts": {"claim": {"claim_id": "claim", "images": []}},
        "vision": {"observations": [{"mapped_damage_type": "HOOD"}]},
    }
    images = import_vision(
        tmp_path, SimpleNamespace(vision_required=False), observation, sections, datetime.now(timezone.utc)
    )
    assert images == {}
    assert sections["vision"].status == status
    assert sections["vision"].data == {"claim_id": "claim", "images": []}
    assert sections["visual_consistency"].status == status
    assert sections["visual_consistency"].data is None


@pytest.mark.parametrize("execution,status", [("FAIL", "FAILED"), ("SKIP", "UNAVAILABLE")])
def test_failed_rag_generation_cannot_present_old_answer(execution, status):
    assessment = ReviewSection(
        status="AVAILABLE", data={"answer": "Old answer", "sources": [], "insufficient_evidence": True}
    )
    result = knowledge_state(assessment, {"components": {"rag_generation": {"status": execution}}})
    assert result.status == status
    assert result.data["answer"] is None and result.data["insufficient_evidence"] is None


def test_insufficient_flag_never_overrides_failed_retrieval():
    assessment = ReviewSection(status="FAILED", data=None)
    assert knowledge_state(assessment, {}).status == "FAILED"
