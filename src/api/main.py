"""FastAPI application factory and public app configuration."""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock

from fastapi import FastAPI
from sqlalchemy.orm import Session, sessionmaker

from api.model_routes import register_investigation_routes, register_model_routes
from api.review import install_review_routes
from api.routes import install_feature_routes, register_data_routes, register_error_handlers
from api.version import application_version
from ml.anomaly_inference import RegisteredAnomalyPredictor, load_candidate_anomaly_predictor
from ml.inference import RegisteredTriagePredictor, load_candidate_predictor
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory

__all__ = ["AppConfig", "app", "create_app"]


@dataclass
class AppConfig:
    """Application service loaders and persistence configuration."""

    predictor_loader: Callable[[], RegisteredTriagePredictor] = load_candidate_predictor
    anomaly_predictor_loader: Callable[[], RegisteredAnomalyPredictor] = load_candidate_anomaly_predictor
    session_factory: sessionmaker[Session] | None = None
    similar_service_loader: Callable | None = None
    knowledge_retriever_loader: Callable | None = None
    knowledge_rag_loader: Callable | None = None
    knowledge_claim_loader: Callable | None = None
    agent_service_loader: Callable | None = None
    agent_claim_loader: Callable | None = None
    agent_run_store_loader: Callable | None = None
    vision_store_loader: Callable | None = None
    review_service_loader: Callable | None = None


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the application with the configured service loaders."""
    config = config or AppConfig()
    app = FastAPI(
        title="Insurance Claims AI Assessment API",
        version=application_version(),
        lifespan=_lifespan(config),
    )
    register_error_handlers(app)
    register_data_routes(app, config)
    register_model_routes(app, config)
    register_investigation_routes(app, config)
    install_feature_routes(app, config)
    install_review_routes(app, service_loader=config.review_service_loader)
    return app


def _lifespan(config: AppConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _initialise_state(app, config)
        engine = _start_database(app, config)
        try:
            yield
        finally:
            _close_services(app, engine)

    return lifespan


def _initialise_state(app, config):
    app.state.triage_predictor = None
    app.state.anomaly_predictor = None
    app.state.model_lock = Lock()
    app.state.similar_service = None
    app.state.knowledge_retriever = None
    app.state.knowledge_rag = None
    app.state.agent_service = None
    app.state.agent_run_store = None
    app.state.vision_store = None
    app.state.review_service = None
    app.state.session_factory = config.session_factory


def _start_database(app, config):
    if config.session_factory is None and os.environ.get("DATABASE_URL"):
        engine = create_database_engine(DatabaseConfig.from_env())
        app.state.session_factory = create_session_factory(engine)
        return engine
    return None


def _close_services(app, engine):
    for attribute in ("similar_service", "knowledge_retriever"):
        service = getattr(app.state, attribute)
        if service is not None and hasattr(service, "close"):
            service.close()
    if engine is not None:
        engine.dispose()


app = create_app()
