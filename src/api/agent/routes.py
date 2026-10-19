"""Bounded HTTP inputs and routes for advisory claim reviews."""

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from agent.models import AgentRunResult
from agent.service import AgentRunError
from api.agent.dependencies import AgentDependencies, AgentRouteConfig
from persistence.repositories import ClaimNotFoundError


class ClaimAgentReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    objective: str | None = Field(default=None, min_length=1, max_length=2000)


def install_agent_routes(app, *, config: AgentRouteConfig | None = None):
    dependencies = AgentDependencies(config)
    _register_review_routes(app, dependencies)
    _register_run_routes(app, dependencies)


def _register_review_routes(app, dependencies):
    @app.post("/v1/claims/{claim_id}/agent/review", response_model=AgentRunResult)
    def review(claim_id: str, request: Request, payload: ClaimAgentReviewRequest | None = None):
        """
        Provider exceptions may include raw output or connection details.
        """
        dependencies.ensure_claim(request, claim_id)
        service = dependencies.agent_service(request)
        try:
            result = service.review_claim(claim_id, objective=payload.objective if payload else None)
        except AgentRunError as error:
            status = 502 if error.failure_kind == "GROUNDING" else 503
            raise HTTPException(status, {"message": "Agent review failed", "run_id": error.run_id}) from error
        except (ClaimNotFoundError, SQLAlchemyError):
            raise
        except Exception as error:
            raise HTTPException(503, "Agent review failed") from error
        if result.status == "FAILED":
            raise HTTPException(503, {"message": "Agent review failed", "run_id": result.run_id})
        return result


def _register_run_routes(app, dependencies):
    @app.get("/v1/claims/{claim_id}/agent/runs", response_model=list[AgentRunResult])
    def claim_runs(claim_id: str, request: Request, limit: int = Query(default=20, ge=1, le=100)):
        dependencies.ensure_claim(request, claim_id)
        return dependencies.run_store(request).list_for_claim(claim_id, limit=limit)

    @app.get("/v1/agent/runs/{run_id}", response_model=AgentRunResult)
    def get_run(run_id: str, request: Request):
        result = dependencies.run_store(request).get_by_run_id(run_id)
        if result is None:
            raise HTTPException(404, "Agent run does not exist")
        return result
