"""Read-only knowledge routes and PostgreSQL-derived claim filters."""

from datetime import date

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from claims.llm import LlmError
from knowledge.factory import create_knowledge_rag_service, create_knowledge_retriever
from knowledge.models import (
    GroundedAnswer,
    GroundingError,
    KnowledgeFilter,
    KnowledgeSearchResponse,
    KnowledgeSourceType,
)
from persistence.repositories import ClaimNotFoundError, ClaimRepository
from retrieval.models import RetrievalUnavailableError


class ClaimKnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000)
    source_type: KnowledgeSourceType
    language: str | None = Field(default="en", min_length=2, max_length=16)
    limit: int = Field(default=5, ge=1, le=20, strict=True)


class KnowledgeSearchRequest(ClaimKnowledgeSearchRequest):
    product: str | None = Field(default=None, min_length=1, max_length=100)
    source_id: str | None = Field(default=None, min_length=1, max_length=100)
    as_of_date: date | None = None


class ClaimKnowledgeAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    source_type: KnowledgeSourceType
    language: str | None = Field(default="en", min_length=2, max_length=16)
    retrieval_limit: int = Field(default=5, ge=1, le=20, strict=True)


class KnowledgeAnswerRequest(ClaimKnowledgeAnswerRequest):
    product: str | None = Field(default=None, min_length=1, max_length=100)
    source_id: str | None = Field(default=None, min_length=1, max_length=100)
    as_of_date: date | None = None


def load_stored_claim(factory, claim_id):
    """
    End the SQL session before embedding/retrieval or LLM network calls.
    """
    with factory() as session:
        claim = ClaimRepository(session).get_by_claim_id(claim_id)
        if claim is None:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return claim


def install_knowledge_routes(app, *, retriever_loader=None, rag_loader=None, claim_loader=None):
    def retriever(request):
        with request.app.state.model_lock:
            if request.app.state.knowledge_retriever is None:
                try:
                    request.app.state.knowledge_retriever = (retriever_loader or create_knowledge_retriever)()
                except (ValueError, RuntimeError) as error:
                    raise HTTPException(503, "Knowledge service configuration failed") from error
            return request.app.state.knowledge_retriever

    def rag(request):
        service = retriever(request)
        with request.app.state.model_lock:
            if request.app.state.knowledge_rag is None:
                request.app.state.knowledge_rag = (rag_loader or create_knowledge_rag_service)(service)
            return request.app.state.knowledge_rag

    def claim_filters(request, claim_id, source_type):
        """
        Procedure is global; the model has no reliable handling date.
        """
        factory = request.app.state.session_factory
        if factory is None and claim_loader is None:
            raise HTTPException(503, "Database is not configured")
        claim = (claim_loader or load_stored_claim)(factory, claim_id)
        if source_type == KnowledgeSourceType.POLICY:
            return {"product": claim.policy.product, "as_of_date": claim.incident.date}
        return {"product": None, "as_of_date": None}

    @app.exception_handler(RetrievalUnavailableError)
    async def retrieval_unavailable(request, error):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(GroundingError)
    async def grounding_failed(request, error):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(LlmError)
    async def generation_failed(request, error):
        """
        Do not expose raw LLM data in an HTTP error response.
        """
        return JSONResponse(
            status_code=503, content={"detail": "Knowledge generation is unavailable or returned invalid data"}
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_failed(request, error):
        return JSONResponse(status_code=503, content={"detail": "Database operation failed"})

    @app.post("/v1/knowledge/search", response_model=KnowledgeSearchResponse)
    def search(payload: KnowledgeSearchRequest, request: Request):
        data = payload.model_dump()
        limit, query, source_type = data.pop("limit"), data.pop("query"), data.pop("source_type")
        return KnowledgeSearchResponse(
            results=retriever(request).search(query, source_type, filters=KnowledgeFilter(**data), limit=limit)
        )

    @app.post("/v1/knowledge/answer", response_model=GroundedAnswer)
    def answer(payload: KnowledgeAnswerRequest, request: Request):
        try:
            data = payload.model_dump()
            retrieval_limit = data.pop("retrieval_limit")
            question = data.pop("question")
            source_type = data.pop("source_type")
            return rag(request).answer(
                question, source_type, filters=KnowledgeFilter(**data), retrieval_limit=retrieval_limit
            )
        except (ConnectionError, TimeoutError) as error:
            raise HTTPException(503, "Knowledge generation is unavailable") from error

    @app.post("/v1/claims/{claim_id}/knowledge/search", response_model=KnowledgeSearchResponse)
    def claim_search(claim_id: str, payload: ClaimKnowledgeSearchRequest, request: Request):
        data = payload.model_dump()
        limit, query, source_type = data.pop("limit"), data.pop("query"), data.pop("source_type")
        data.update(claim_filters(request, claim_id, source_type))
        return KnowledgeSearchResponse(
            results=retriever(request).search(query, source_type, filters=KnowledgeFilter(**data), limit=limit)
        )

    @app.post("/v1/claims/{claim_id}/knowledge/answer", response_model=GroundedAnswer)
    def claim_answer(claim_id: str, payload: ClaimKnowledgeAnswerRequest, request: Request):
        data = payload.model_dump()
        retrieval_limit = data.pop("retrieval_limit")
        question = data.pop("question")
        source_type = data.pop("source_type")
        data.update(claim_filters(request, claim_id, source_type))
        try:
            return rag(request).answer(
                question, source_type, filters=KnowledgeFilter(**data), retrieval_limit=retrieval_limit
            )
        except (ConnectionError, TimeoutError) as error:
            raise HTTPException(503, "Knowledge generation is unavailable") from error
