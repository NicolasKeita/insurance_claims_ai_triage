from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Self
import os
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from claims.consistency import ClaimConsistencyReport
from investigation.engine import assess_investigation
from investigation.models import InvestigationAssessment
from investigation.policy import DEFAULT_INVESTIGATION_POLICY
from investigation.history import HistoricalClaimProfile
from investigation.service import assess_stored_claim
from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from persistence.history import HistoricalClaimRepository
from persistence.repositories import ClaimNotFoundError
from ml.anomaly_inference import (
    RegisteredAnomalyAssessment,
    RegisteredAnomalyPredictor,
    load_candidate_anomaly_predictor,
)
from ml.anomaly_training import AnomalyAssessment
from ml.features import TriageFeatures, count_consistency_issues
from ml.inference import (
    RegisteredTriagePredictor,
    TriagePrediction,
    load_candidate_predictor,
)

PredictorLoader = Callable[[], RegisteredTriagePredictor]
AnomalyPredictorLoader = Callable[[], RegisteredAnomalyPredictor]


class ClaimsAssessment(BaseModel):
    triage: TriagePrediction
    anomaly: RegisteredAnomalyAssessment
    investigation: InvestigationAssessment


class StructuredEvidenceRequest(BaseModel):
    features: TriageFeatures
    consistency: ClaimConsistencyReport

    @model_validator(mode="after")
    def validate_consistency_counts(self) -> Self:
        if self.features.additional_damage_count != len(
            self.consistency.additional_quote_damage
        ):
            raise ValueError(
                "additional_damage_count must match additional_quote_damage"
            )
        if self.features.consistency_issue_count != count_consistency_issues(
            self.consistency
        ):
            raise ValueError("consistency_issue_count must match consistency report")
        return self


class InvestigationAssessmentRequest(StructuredEvidenceRequest):
    anomaly: AnomalyAssessment


class ClaimsAssessmentRequest(StructuredEvidenceRequest):
    pass


class StoredInvestigationRequest(BaseModel):
    # Only comparisons are supplied: stored claims and history are authoritative.
    model_config = ConfigDict(extra="forbid")
    consistency: ClaimConsistencyReport


def create_app(
    predictor_loader: PredictorLoader = load_candidate_predictor,
    anomaly_predictor_loader: AnomalyPredictorLoader = load_candidate_anomaly_predictor,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.triage_predictor = None
        app.state.anomaly_predictor = None
        app.state.model_lock = Lock()
        engine = None
        app.state.session_factory = session_factory
        if session_factory is None and os.environ.get("DATABASE_URL"):
            engine = create_database_engine(DatabaseConfig.from_env())
            app.state.session_factory = create_session_factory(engine)
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    def predictor(request: Request, kind: str):
        # Read-only history remains usable without loading any MLflow model.
        with request.app.state.model_lock:
            attribute = f"{kind}_predictor"
            value = getattr(request.app.state, attribute)
            if value is None:
                loader = predictor_loader if kind == "triage" else anomaly_predictor_loader
                value = loader()
                setattr(request.app.state, attribute, value)
            return value

    def database_session(request: Request):
        factory = request.app.state.session_factory
        if factory is None:
            raise HTTPException(status_code=503, detail="Database is not configured")
        try:
            # One fresh session/transaction, with rollback on any exception.
            with factory.begin() as session:
                yield session
        except SQLAlchemyError as error:
            raise HTTPException(status_code=503, detail="Database operation failed") from error

    app = FastAPI(
        title="Insurance Claims AI Assessment API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ClaimNotFoundError)
    async def claim_not_found(request: Request, error: ClaimNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get("/v1/claims/{claim_id}/history", response_model=HistoricalClaimProfile)
    def historical_profile(claim_id: str, session: Session = Depends(database_session, scope="function")):
        return HistoricalClaimRepository(session).get_profile(claim_id)

    @app.post("/v1/claims/{claim_id}/investigation/assess", response_model=InvestigationAssessment)
    def stored_investigation(
        claim_id: str, payload: StoredInvestigationRequest, request: Request,
        session: Session = Depends(database_session, scope="function"),
    ):
        return assess_stored_claim(
            session, claim_id, payload.consistency,
            lambda features: predictor(request, "anomaly").assess(features),
        )

    @app.get("/health")
    def health(request: Request):
        triage = predictor(request, "triage")
        anomaly = predictor(request, "anomaly")
        return {
            "status": "ok",
            "models": {
                "triage": {
                    "name": triage.registered_model,
                    "version": triage.version,
                    "alias": triage.alias,
                },
                "anomaly": {
                    "name": anomaly.registered_model,
                    "version": anomaly.version,
                    "alias": anomaly.alias,
                },
            },
            "investigation_policy_version": DEFAULT_INVESTIGATION_POLICY.version,
        }

    @app.post("/v1/triage/predict", response_model=TriagePrediction)
    def predict_triage(features: TriageFeatures, request: Request) -> TriagePrediction:
        return predictor(request, "triage").predict(features)

    @app.post("/v1/anomaly/assess", response_model=RegisteredAnomalyAssessment)
    def assess_anomaly(
        features: TriageFeatures, request: Request
    ) -> RegisteredAnomalyAssessment:
        return predictor(request, "anomaly").assess(features)

    @app.post("/v1/investigation/assess", response_model=InvestigationAssessment)
    def assess_investigation_request(
        payload: InvestigationAssessmentRequest,
    ) -> InvestigationAssessment:
        return assess_investigation(
            consistency=payload.consistency,
            anomaly=payload.anomaly,
            features=payload.features,
        )

    @app.post("/v1/claims/assess", response_model=ClaimsAssessment)
    def assess_claim(
        payload: ClaimsAssessmentRequest, request: Request
    ) -> ClaimsAssessment:
        triage = predictor(request, "triage").predict(payload.features)
        anomaly = predictor(request, "anomaly").assess(payload.features)
        return ClaimsAssessment(
            triage=triage,
            anomaly=anomaly,
            investigation=assess_investigation(
                consistency=payload.consistency,
                anomaly=anomaly.assessment,
                features=payload.features,
            ),
        )

    return app


app = create_app()
