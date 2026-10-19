"""Data and feature routes, HTTP error handlers and request-scoped helpers."""

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.agent import AgentRouteConfig, install_agent_routes
from api.knowledge import install_knowledge_routes
from api.vision import install_vision_routes
from investigation.history import HistoricalClaimProfile
from persistence.history import HistoricalClaimRepository
from persistence.repositories import ClaimNotFoundError
from retrieval.config import DEFAULT_LIMIT, MAX_LIMIT
from retrieval.models import RetrievalUnavailableError, SimilarClaimsResponse
from retrieval.similar_claims import create_similar_claims_service


def predictor(request: Request, kind: str, config):
    """
    Read-only history remains usable without loading any MLflow model.
    """
    with request.app.state.model_lock:
        attribute = f"{kind}_predictor"
        value = getattr(request.app.state, attribute)
        if value is None:
            loader = config.predictor_loader if kind == "triage" else config.anomaly_predictor_loader
            value = loader()
            setattr(request.app.state, attribute, value)
        return value


def database_session(request: Request):
    """
    One fresh session/transaction, with rollback on any exception.
    """
    factory = request.app.state.session_factory
    if factory is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
    try:
        with factory.begin() as session:
            yield session
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail="Database operation failed") from error


def register_error_handlers(app):
    @app.exception_handler(ClaimNotFoundError)
    async def claim_not_found(request: Request, error: ClaimNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(error)})


def register_data_routes(app, config):
    @app.get("/v1/claims/{claim_id}/history", response_model=HistoricalClaimProfile)
    def historical_profile(claim_id: str, session: Session = Depends(database_session, scope="function")):
        return HistoricalClaimRepository(session).get_profile(claim_id)

    @app.get("/v1/claims/{claim_id}/similar", response_model=SimilarClaimsResponse)
    def similar_claims(
        claim_id: str,
        request: Request,
        limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ):
        try:
            with request.app.state.model_lock:
                if request.app.state.similar_service is None:
                    factory = request.app.state.session_factory
                    if config.similar_service_loader is None and factory is None:
                        raise HTTPException(status_code=503, detail="Database is not configured")
                    try:
                        request.app.state.similar_service = (
                            config.similar_service_loader(factory)
                            if config.similar_service_loader is not None
                            else create_similar_claims_service(factory)
                        )
                    except (ValueError, RuntimeError) as error:
                        raise HTTPException(
                            status_code=503, detail="Similar claims service configuration failed"
                        ) from error
                service = request.app.state.similar_service
            return service.search(claim_id, limit)
        except RetrievalUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(status_code=503, detail="Database operation failed") from error


def install_feature_routes(app, config):
    install_knowledge_routes(
        app,
        retriever_loader=config.knowledge_retriever_loader,
        rag_loader=config.knowledge_rag_loader,
        claim_loader=config.knowledge_claim_loader,
    )
    install_agent_routes(
        app,
        config=AgentRouteConfig(
            service_loader=config.agent_service_loader,
            claim_loader=config.agent_claim_loader,
            run_store_loader=config.agent_run_store_loader,
            similar_service_loader=config.similar_service_loader,
            knowledge_retriever_loader=config.knowledge_retriever_loader,
        ),
    )
    install_vision_routes(app, store_loader=config.vision_store_loader)
