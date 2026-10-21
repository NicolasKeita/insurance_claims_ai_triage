"""Knowledge search and answer routes with PostgreSQL-derived claim filters."""

from datetime import date

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from api.knowledge.dependencies import KnowledgeDependencies, register_knowledge_exception_handlers
from retrieval.knowledge.models import GroundedAnswer, KnowledgeFilter, KnowledgeSearchResponse, KnowledgeSourceType


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


def install_knowledge_routes(app, *, retriever_loader=None, rag_loader=None, claim_loader=None):
    dependencies = KnowledgeDependencies(retriever_loader, rag_loader, claim_loader)
    register_knowledge_exception_handlers(app)
    _register_knowledge_routes(app, dependencies)
    _register_claim_knowledge_routes(app, dependencies)


def _register_knowledge_routes(app, dependencies):
    @app.post("/v1/knowledge/search", response_model=KnowledgeSearchResponse)
    def search(payload: KnowledgeSearchRequest, request: Request):
        data = payload.model_dump()
        limit, query, source_type = data.pop("limit"), data.pop("query"), data.pop("source_type")
        return KnowledgeSearchResponse(
            results=dependencies.retriever(request).search(
                query, source_type, filters=KnowledgeFilter(**data), limit=limit
            )
        )

    @app.post("/v1/knowledge/answer", response_model=GroundedAnswer)
    def answer(payload: KnowledgeAnswerRequest, request: Request):
        return _answer(dependencies, request, payload.model_dump())


def _register_claim_knowledge_routes(app, dependencies):
    @app.post("/v1/claims/{claim_id}/knowledge/search", response_model=KnowledgeSearchResponse)
    def claim_search(claim_id: str, payload: ClaimKnowledgeSearchRequest, request: Request):
        data = payload.model_dump()
        limit, query, source_type = data.pop("limit"), data.pop("query"), data.pop("source_type")
        data.update(dependencies.claim_filters(request, claim_id, source_type))
        return KnowledgeSearchResponse(
            results=dependencies.retriever(request).search(
                query, source_type, filters=KnowledgeFilter(**data), limit=limit
            )
        )

    @app.post("/v1/claims/{claim_id}/knowledge/answer", response_model=GroundedAnswer)
    def claim_answer(claim_id: str, payload: ClaimKnowledgeAnswerRequest, request: Request):
        data = payload.model_dump()
        data.update(dependencies.claim_filters(request, claim_id, data["source_type"]))
        return _answer(dependencies, request, data)


def _answer(dependencies, request, data):
    try:
        retrieval_limit = data.pop("retrieval_limit")
        question = data.pop("question")
        source_type = data.pop("source_type")
        return dependencies.rag(request).answer(
            question, source_type, filters=KnowledgeFilter(**data), retrieval_limit=retrieval_limit
        )
    except (ConnectionError, TimeoutError) as error:
        raise HTTPException(503, "Knowledge generation is unavailable") from error
