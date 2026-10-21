"""Model and investigation assessment routes."""

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from api.routes import database_session, predictor
from api.schemas import (
    ClaimsAssessment,
    ClaimsAssessmentRequest,
    InvestigationAssessmentRequest,
    StoredInvestigationRequest,
)
from claims.investigation.engine import assess_investigation
from claims.investigation.models import InvestigationAssessment
from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY
from claims.investigation.service import assess_stored_claim
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.features import TriageFeatures
from ml.inference import TriagePrediction


def register_model_routes(app, config):
    @app.get("/health")
    def health(request: Request):
        triage = predictor(request, "triage", config)
        anomaly = predictor(request, "anomaly", config)
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
        return predictor(request, "triage", config).predict(features)

    @app.post("/v1/anomaly/assess", response_model=RegisteredAnomalyAssessment)
    def assess_anomaly(features: TriageFeatures, request: Request) -> RegisteredAnomalyAssessment:
        return predictor(request, "anomaly", config).assess(features)


def register_investigation_routes(app, config):
    @app.post("/v1/claims/{claim_id}/investigation/assess", response_model=InvestigationAssessment)
    def stored_investigation(
        claim_id: str,
        payload: StoredInvestigationRequest,
        request: Request,
        session: Session = Depends(database_session, scope="function"),
    ):
        return assess_stored_claim(
            session,
            claim_id,
            payload.consistency,
            lambda features: predictor(request, "anomaly", config).assess(features),
        )

    @app.post("/v1/investigation/assess", response_model=InvestigationAssessment)
    def assess_investigation_request(payload: InvestigationAssessmentRequest) -> InvestigationAssessment:
        return assess_investigation(
            consistency=payload.consistency,
            anomaly=payload.anomaly,
            features=payload.features,
        )

    @app.post("/v1/claims/assess", response_model=ClaimsAssessment)
    def assess_claim(payload: ClaimsAssessmentRequest, request: Request) -> ClaimsAssessment:
        triage = predictor(request, "triage", config).predict(payload.features)
        anomaly = predictor(request, "anomaly", config).assess(payload.features)
        return ClaimsAssessment(
            triage=triage,
            anomaly=anomaly,
            investigation=assess_investigation(
                consistency=payload.consistency,
                anomaly=anomaly.assessment,
                features=payload.features,
            ),
        )
