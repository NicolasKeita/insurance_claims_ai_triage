"""Lifespan-cached knowledge services and shared HTTP error handlers."""

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from claims.llm import LlmError
from persistence.repositories import ClaimNotFoundError, ClaimRepository
from retrieval.knowledge.factory import create_knowledge_rag_service, create_knowledge_retriever
from retrieval.knowledge.models import GroundingError, KnowledgeSourceType
from retrieval.models import RetrievalUnavailableError


def load_stored_claim(factory, claim_id):
    """
    End the SQL session before embedding/retrieval or LLM network calls.
    """
    with factory() as session:
        claim = ClaimRepository(session).get_by_claim_id(claim_id)
        if claim is None:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
        return claim


class KnowledgeDependencies:
    """Resolve knowledge retrieval, generation and claim filters lazily."""

    def __init__(self, retriever_loader=None, rag_loader=None, claim_loader=None):
        self.retriever_loader = retriever_loader
        self.rag_loader = rag_loader
        self.claim_loader = claim_loader

    def retriever(self, request):
        with request.app.state.model_lock:
            if request.app.state.knowledge_retriever is None:
                loader = self.retriever_loader or create_knowledge_retriever
                try:
                    request.app.state.knowledge_retriever = loader()
                except (ValueError, RuntimeError) as error:
                    raise HTTPException(503, "Knowledge service configuration failed") from error
            return request.app.state.knowledge_retriever

    def rag(self, request):
        service = self.retriever(request)
        with request.app.state.model_lock:
            if request.app.state.knowledge_rag is None:
                request.app.state.knowledge_rag = (self.rag_loader or create_knowledge_rag_service)(service)
            return request.app.state.knowledge_rag

    def claim_filters(self, request, claim_id, source_type):
        """
        Procedure is global; the model has no reliable handling date.
        """
        factory = request.app.state.session_factory
        if factory is None and self.claim_loader is None:
            raise HTTPException(503, "Database is not configured")
        claim = (self.claim_loader or load_stored_claim)(factory, claim_id)
        if source_type == KnowledgeSourceType.POLICY:
            return {"product": claim.policy.product, "as_of_date": claim.incident.date}
        return {"product": None, "as_of_date": None}


def register_knowledge_exception_handlers(app):
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
