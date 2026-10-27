"""Read-only claim review routes in the existing FastAPI application."""

import os

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError

from claims.review.models import ClaimReview, DemoClaim
from claims.review.persisted import PersistedReviewService


def install_review_routes(app, *, service_loader=None):
    @app.get("/v1/review/demo-claims", response_model=list[DemoClaim])
    def demo_claims(request: Request):
        return _read_service(request, service_loader, "list_demo_claims")

    @app.get("/v1/claims/{claim_id}/review", response_model=ClaimReview)
    def review(claim_id: str, request: Request):
        return _read_service(request, service_loader, "get_review", claim_id)

    @app.get("/v1/claims/{claim_id}/review/images/{filename}", response_class=FileResponse)
    def image(claim_id: str, filename: str, request: Request):
        if filename in {".", ".."} or any(char in filename for char in "/\\:"):
            raise HTTPException(404, "The original claim photograph is unavailable")
        path = _read_service(request, service_loader, "image_path", claim_id, filename)
        if path is None:
            raise HTTPException(404, "The original claim photograph is unavailable")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=60", "X-Content-Type-Options": "nosniff"})


def _read_service(request, loader, method, *args):
    try:
        return getattr(_service(request, loader), method)(*args)
    except (SQLAlchemyError, OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(503, "Stored claim review data is temporarily unavailable") from error


def _service(request, loader):
    with request.app.state.model_lock:
        service = request.app.state.review_service
        if service is None:
            factory = request.app.state.session_factory
            if loader is not None:
                service = loader(factory)
            elif os.environ.get("CLAIMS_REVIEW_DEMO") == "1":
                from claims.review.demo import DemoReviewService

                try:
                    service = DemoReviewService()
                except (OSError, ValueError) as error:
                    raise HTTPException(
                        503, "Demo snapshots are unavailable; run scripts/prepare_review_demo.py first"
                    ) from error
            elif factory is not None:
                service = PersistedReviewService(factory)
            else:
                raise HTTPException(503, "Claim review requires configured persistence or prepared demo snapshots")
            request.app.state.review_service = service
        return service
