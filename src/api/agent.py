"""Bounded HTTP inputs and lifespan-cached dependencies for advisory reviews."""

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from agent.factory import create_claims_agent_service
from agent.models import AgentRunResult
from persistence.agent_runs import AgentRunStore
from agent.service import AgentRunError
from api.knowledge import load_stored_claim
from claims.llm import LlmError
from knowledge.factory import create_knowledge_retriever
from retrieval.similar_claims import create_similar_claims_service
from persistence.repositories import ClaimNotFoundError
from sqlalchemy.exc import SQLAlchemyError


class ClaimAgentReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    objective: str | None = Field(default=None, min_length=1, max_length=2000)


def install_agent_routes(
    app, *, service_loader=None, claim_loader=None, run_store_loader=None,
    similar_service_loader=None, knowledge_retriever_loader=None,
):
    def ensure_claim(request, claim_id):
        factory = request.app.state.session_factory
        if factory is None and claim_loader is None:
            raise HTTPException(503, "Database is not configured")
        # A nonexistent claim must fail before any LLM/embedding model is loaded.
        return (claim_loader or load_stored_claim)(factory, claim_id)

    def similar_service(request):
        with request.app.state.model_lock:
            if request.app.state.similar_service is None:
                factory = request.app.state.session_factory
                request.app.state.similar_service = (
                    (similar_service_loader or create_similar_claims_service)(factory)
                )
            return request.app.state.similar_service

    def knowledge_retriever(request):
        with request.app.state.model_lock:
            if request.app.state.knowledge_retriever is None:
                request.app.state.knowledge_retriever = (
                    (knowledge_retriever_loader or create_knowledge_retriever)()
                )
            return request.app.state.knowledge_retriever

    def agent_service(request):
        with request.app.state.model_lock:
            if request.app.state.agent_service is None:
                try:
                    request.app.state.agent_service = (service_loader or create_claims_agent_service)(
                        request.app.state.session_factory,
                        similar_service=request.app.state.similar_service,
                        knowledge_retriever=request.app.state.knowledge_retriever,
                        similar_service_loader=lambda: similar_service(request),
                        knowledge_retriever_loader=lambda: knowledge_retriever(request),
                    )
                except (ValueError, RuntimeError, LlmError) as error:
                    raise HTTPException(503, "Agent service configuration failed") from error
            return request.app.state.agent_service

    def run_store(request):
        with request.app.state.model_lock:
            if request.app.state.agent_run_store is None:
                factory = request.app.state.session_factory
                if factory is None and run_store_loader is None:
                    raise HTTPException(503, "Database is not configured")
                request.app.state.agent_run_store = (run_store_loader or AgentRunStore)(factory)
            return request.app.state.agent_run_store

    @app.post("/v1/claims/{claim_id}/agent/review", response_model=AgentRunResult)
    def review(
        claim_id: str, request: Request, payload: ClaimAgentReviewRequest | None = None,
    ):
        ensure_claim(request, claim_id)
        service = agent_service(request)
        try:
            result = service.review_claim(claim_id, objective=payload.objective if payload else None)
        except AgentRunError as error:
            status = 502 if error.failure_kind == "GROUNDING" else 503
            raise HTTPException(status, {"message": "Agent review failed", "run_id": error.run_id}) from error
        except (ClaimNotFoundError, SQLAlchemyError):
            raise
        except Exception as error:
            # Provider exceptions may include raw output or connection details.
            raise HTTPException(503, "Agent review failed") from error
        if result.status == "FAILED":
            raise HTTPException(503, {"message": "Agent review failed", "run_id": result.run_id})
        return result

    @app.get("/v1/claims/{claim_id}/agent/runs", response_model=list[AgentRunResult])
    def claim_runs(
        claim_id: str, request: Request, limit: int = Query(default=20, ge=1, le=100),
    ):
        ensure_claim(request, claim_id)
        return run_store(request).list_for_claim(claim_id, limit=limit)

    @app.get("/v1/agent/runs/{run_id}", response_model=AgentRunResult)
    def get_run(run_id: str, request: Request):
        result = run_store(request).get_by_run_id(run_id)
        if result is None:
            raise HTTPException(404, "Agent run does not exist")
        return result
