"""Compose the advisory agent from existing read-only services and structured LLM."""

import os

from claims.llm import LlmError, OllamaStructuredLlm
from agent.config import AGENT_LLM_CONTEXT_WINDOW

from persistence.agent_runs import AgentRunStore
from agent.planner import AGENT_SYSTEM_PROMPT, StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import ClaimsAgentService
from agent.tools import ReadOnlyAgentTools


def create_claims_agent_service(
    factory, *, similar_service=None, knowledge_retriever=None,
    similar_service_loader=None, knowledge_retriever_loader=None,
):
    """Construct once; retrieval components are loaded only when a tool needs them.

    Injected loader callbacks let FastAPI reuse its lifespan-scoped caches.
    The standalone script gets the same lazy behavior without constructing an app.
    """
    if factory is None:
        raise ValueError("Database is not configured")
    model = os.environ.get("CLAIMS_LLM_MODEL")
    if not model:
        raise LlmError("CLAIMS_LLM_MODEL is required for agent generation")

    if similar_service is None and similar_service_loader is None:
        from retrieval.similar_claims import create_similar_claims_service

        similar_service_loader = lambda: create_similar_claims_service(factory)
    if knowledge_retriever is None and knowledge_retriever_loader is None:
        from knowledge.factory import create_knowledge_retriever

        knowledge_retriever_loader = create_knowledge_retriever
    llm = OllamaStructuredLlm(
        model=model,
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        system_prompt=AGENT_SYSTEM_PROMPT,
        context_window=AGENT_LLM_CONTEXT_WINDOW,
    )
    tools = ReadOnlyAgentTools(
        factory,
        similar_service=similar_service,
        knowledge_retriever=knowledge_retriever,
        similar_service_loader=similar_service_loader,
        knowledge_retriever_loader=knowledge_retriever_loader,
    )
    return ClaimsAgentService(
        tools, StructuredAgentPlanner(llm), StructuredAgentFinalizer(llm),
        run_store=AgentRunStore(factory), llm_model=model,
    )
