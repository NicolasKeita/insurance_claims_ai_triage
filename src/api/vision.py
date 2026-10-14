"""Read computed vision snapshots without loading or rerunning either detector.

The stored Claim has image filenames but no trusted filesystem root. Inference
remains an explicit ClaimCase operation until that root is configured centrally.
"""

from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel

from persistence.vision_assessments import VisionAssessmentStore
from vision.models import ClaimMultimodalVisionAssessment


class StoredVisionAssessmentResponse(BaseModel):
    assessment_id: UUID
    created_at: datetime
    result: ClaimMultimodalVisionAssessment


def install_vision_routes(app, *, store_loader=None):
    @app.get("/v1/claims/{claim_id}/vision/latest", response_model=StoredVisionAssessmentResponse)
    def latest_visual_assessment(claim_id: str, request: Request):
        with request.app.state.model_lock:
            store = request.app.state.vision_store
            if store is None:
                factory = request.app.state.session_factory
                if factory is None and store_loader is None:
                    raise HTTPException(503, "Database is not configured")
                store = (store_loader or VisionAssessmentStore)(factory)
                request.app.state.vision_store = store
        record = store.latest_for_claim(claim_id)
        if record is None:
            raise HTTPException(404, "No stored visual assessment is available")
        return StoredVisionAssessmentResponse(
            assessment_id=record.id, created_at=record.created_at, result=record.result,
        )
