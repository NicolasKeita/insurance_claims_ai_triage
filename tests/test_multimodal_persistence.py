"""JSONB domain mapping without a database or checkpoint."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from test_multimodal_investigation import make_visual_assessment

from persistence.vision_assessments import (
    vision_assessment_from_row,
    vision_assessment_to_row,
)


def test_vision_snapshot_round_trip_omits_computed_geometry_duplicates():
    visual = make_visual_assessment()
    row = vision_assessment_to_row(uuid4(), visual)
    row.id, row.created_at = uuid4(), datetime.now(timezone.utc)
    assert "observations" not in row.structured_results
    assert "aggregate_mapped_damage_types" not in row.structured_results
    assert "fusion_policy_version" not in row.structured_results
    assert "filename" not in row.structured_results["images"][0]
    assert vision_assessment_from_row(row).result == visual
    assert row.damage_model_version == "damage-v1"
    assert row.part_model_version == "parts-v1"
    assert row.fusion_policy_version == "vision_fusion_v1"
    json.dumps(row.structured_results, allow_nan=False)
    visual.images[0].damage_assessment.detections.clear()
    assert len(row.structured_results["images"][0]["damage_assessment"]["detections"]) == 1


def test_row_metadata_corruption_cannot_silently_replace_snapshot_metadata():
    row = vision_assessment_to_row(uuid4(), make_visual_assessment())
    row.id, row.created_at = uuid4(), datetime.now(timezone.utc)
    row.part_model_version = "different-version"
    with pytest.raises(ValueError, match="metadata differs"):
        vision_assessment_from_row(row)


def test_embedded_claim_id_must_match_the_claim_selected_by_the_repository():
    row = vision_assessment_to_row(uuid4(), make_visual_assessment(claim_id="CLAIM-OTHER"))
    row.id, row.created_at = uuid4(), datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="different claim"):
        vision_assessment_from_row(row, expected_claim_id="CLAIM-2026-00001")
