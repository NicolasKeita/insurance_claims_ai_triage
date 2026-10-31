"""Small frozen review, with independent component IDs and source files."""

from datetime import datetime, timezone

from claims.review.assembly import build_review
from claims.review.models import ReviewSection
from tests.claims.review_archive._assets import IMAGE, PDF, claim_inventory, source_assets

__all__ = ["IMAGE", "PDF", "recorded_review", "source_assets"]

CAPTURED = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)


def recorded_review():
    sections = {"investigation": _investigation(), "vision": _vision(), "agent": _agent()}
    review = build_review(claim_inventory(), sections)
    review.generated_at = CAPTURED
    return review


def _investigation():
    return ReviewSection(
        status="AVAILABLE",
        source="investigation:frozen-001",
        updated_at=CAPTURED,
        data={
            "review_priority": "NORMAL",
            "review_score": 12,
            "signals": [
                {
                    "code": "VISUAL_ADDITIONAL_DAMAGE",
                    "title": "Recorded hood damage",
                    "explanation": "Hood damage needs human confirmation.",
                    "severity": "MEDIUM",
                }
            ],
        },
    )


def _vision():
    return ReviewSection(
        status="AVAILABLE",
        source="vision:frozen-002",
        updated_at=CAPTURED,
        data={
            "images": [
                {
                    "filename": "photo originale.png",
                    "image_url": "/v1/claims/CLAIM-0001/review/images/photo%20originale.png",
                    "evidence": [],
                }
            ]
        },
    )


def _agent():
    return ReviewSection(
        status="AVAILABLE",
        source="agent:stable-run-003",
        updated_at=CAPTURED,
        data={
            "status": "COMPLETED",
            "run_id": "stable-run-003",
            "agent_version": "recorded-agent-v1",
            "prompt_version": "recorded-prompt-v2",
            "recommendation": {
                "case_summary": "Only the recorded hood finding was assessed.",
                "recommended_next_action": "REQUEST_ADDITIONAL_INFORMATION",
                "rationale": [
                    {
                        "statement": "The photograph supports requesting clarification.",
                        "evidence_ids": ["investigation:frozen-001"],
                    }
                ],
                "evidence_ids": ["investigation:frozen-001"],
                "uncertainties": ["The repair requires a human decision."],
                "human_review_required": True,
            },
            "tools_used": [],
        },
    )
