"""Lifespan-cached dependency loaders for the advisory review routes."""

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException

from agent.factory import create_claims_agent_service
from api.knowledge import load_stored_claim
from claims.llm import LlmError
from knowledge.factory import create_knowledge_retriever
from persistence.agent_runs import AgentRunStore
from retrieval.similar_claims import create_similar_claims_service


@dataclass
class AgentRouteConfig:
    """Service loaders used by claim review routes."""

    service_loader: Callable | None = None
    claim_loader: Callable | None = None
    run_store_loader: Callable | None = None
    similar_service_loader: Callable | None = None
    knowledge_retriever_loader: Callable | None = None


class AgentDependencies:
    """Resolve claim, review and retrieval services with lifespan caching."""

    def __init__(self, config: AgentRouteConfig | None = None):
        config = config or AgentRouteConfig()
        self.service_loader = config.service_loader
        self.claim_loader = config.claim_loader
        self.run_store_loader = config.run_store_loader
        self.similar_service_loader = config.similar_service_loader
        self.knowledge_retriever_loader = config.knowledge_retriever_loader

    def ensure_claim(self, request, claim_id):
        """A nonexistent claim must fail before any LLM/embedding model is loaded."""
        factory = request.app.state.session_factory
        if factory is None and self.claim_loader is None:
            raise HTTPException(503, "Database is not configured")
        return (self.claim_loader or load_stored_claim)(factory, claim_id)

    def similar_service(self, request):
        with request.app.state.model_lock:
            if request.app.state.similar_service is None:
                factory = request.app.state.session_factory
                loader = self.similar_service_loader or create_similar_claims_service
                request.app.state.similar_service = loader(factory)
            return request.app.state.similar_service

    def knowledge_retriever(self, request):
        with request.app.state.model_lock:
            if request.app.state.knowledge_retriever is None:
                loader = self.knowledge_retriever_loader or create_knowledge_retriever
                request.app.state.knowledge_retriever = loader()
            return request.app.state.knowledge_retriever

    def agent_service(self, request):
        with request.app.state.model_lock:
            if request.app.state.agent_service is None:
                self._create_agent_service(request)
            return request.app.state.agent_service

    def _create_agent_service(self, request):
        loader = self.service_loader or create_claims_agent_service
        try:
            request.app.state.agent_service = loader(
                request.app.state.session_factory,
                similar_service=request.app.state.similar_service,
                knowledge_retriever=request.app.state.knowledge_retriever,
                similar_service_loader=lambda: self.similar_service(request),
                knowledge_retriever_loader=lambda: self.knowledge_retriever(request),
            )
        except (ValueError, RuntimeError, LlmError) as error:
            raise HTTPException(503, "Agent service configuration failed") from error

    def run_store(self, request):
        with request.app.state.model_lock:
            if request.app.state.agent_run_store is None:
                factory = request.app.state.session_factory
                if factory is None and self.run_store_loader is None:
                    raise HTTPException(503, "Database is not configured")
                request.app.state.agent_run_store = (self.run_store_loader or AgentRunStore)(factory)
            return request.app.state.agent_run_store
