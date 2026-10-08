from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from ml.features import TriageFeatures
from ml.inference import (
    RegisteredTriagePredictor,
    TriagePrediction,
    load_candidate_predictor,
)

PredictorLoader = Callable[[], RegisteredTriagePredictor]


def create_app(predictor_loader: PredictorLoader = load_candidate_predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor_loader()
        yield

    app = FastAPI(
        title="Insurance Claims AI Triage API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        predictor = request.app.state.predictor
        return {
            "status": "ok",
            "model": predictor.registered_model,
            "version": predictor.version,
            "alias": predictor.alias,
        }

    @app.post("/v1/triage/predict", response_model=TriagePrediction)
    def predict_triage(features: TriageFeatures, request: Request) -> TriagePrediction:
        return request.app.state.predictor.predict(features)

    return app


app = create_app()
