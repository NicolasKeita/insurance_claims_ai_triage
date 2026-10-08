from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from ml.anomaly_inference import (
    RegisteredAnomalyAssessment,
    RegisteredAnomalyPredictor,
    load_candidate_anomaly_predictor,
)
from ml.features import TriageFeatures
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


def create_app(
    predictor_loader: PredictorLoader = load_candidate_predictor,
    anomaly_predictor_loader: AnomalyPredictorLoader = load_candidate_anomaly_predictor,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.triage_predictor = predictor_loader()
        app.state.anomaly_predictor = anomaly_predictor_loader()
        yield

    app = FastAPI(
        title="Insurance Claims AI Assessment API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        triage = request.app.state.triage_predictor
        anomaly = request.app.state.anomaly_predictor
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
        }

    @app.post("/v1/triage/predict", response_model=TriagePrediction)
    def predict_triage(features: TriageFeatures, request: Request) -> TriagePrediction:
        return request.app.state.triage_predictor.predict(features)

    @app.post("/v1/anomaly/assess", response_model=RegisteredAnomalyAssessment)
    def assess_anomaly(
        features: TriageFeatures, request: Request
    ) -> RegisteredAnomalyAssessment:
        return request.app.state.anomaly_predictor.assess(features)

    @app.post("/v1/claims/assess", response_model=ClaimsAssessment)
    def assess_claim(features: TriageFeatures, request: Request) -> ClaimsAssessment:
        return ClaimsAssessment(
            triage=request.app.state.triage_predictor.predict(features),
            anomaly=request.app.state.anomaly_predictor.assess(features),
        )

    return app


app = create_app()
