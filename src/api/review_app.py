"""Read-only archive console API, independent of inference and database clients."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.review import install_review_routes
from api.version import application_version
from claims.review.demo import DemoReviewService
from persistence.repositories.errors import ClaimNotFoundError


def create_app(archive_root: str | Path | None = None) -> FastAPI:
    root = archive_root or os.environ.get("CLAIMS_REVIEW_DEMO_DIR", "analysis_results")
    app = FastAPI(
        title="Insurance Claims Read-only Review API",
        version=application_version(),
        lifespan=_lifespan(root),
    )
    install_review_routes(app)

    @app.exception_handler(ClaimNotFoundError)
    async def claim_not_found(request: Request, error: ClaimNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get("/readyz")
    def readiness():
        return {"status": "ready", "mode": "archive", "claims": len(app.state.review_service.list_demo_claims())}

    return app


def _lifespan(root):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model_lock = Lock()
        app.state.session_factory = None
        app.state.review_service = _load_archives(root)
        yield

    return lifespan


def _load_archives(root):
    try:
        service = DemoReviewService(root)
        claims = service.list_demo_claims()
        if not claims:
            raise ValueError("The archive catalogue is empty")
        for claim in claims:
            service.get_review(claim.claim_id)
        return service
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            "Cannot start Claim Review: mount an existing, non-empty analysis_results directory "
            "with a valid manifest.json at CLAIMS_REVIEW_DEMO_DIR. No analysis was generated."
        ) from error


app = create_app()
