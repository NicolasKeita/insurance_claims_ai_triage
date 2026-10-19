"""Compose the advisory agent from existing read-only services and structured LLM."""

import os

from agent.config import AGENT_LLM_CONTEXT_WINDOW
from agent.planner import AGENT_SYSTEM_PROMPT, StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentServiceConfig, ClaimsAgentService
from agent.tools import ReadOnlyAgentTools
from claims.llm import LlmError, OllamaStructuredLlm
from persistence.agent_runs import AgentRunStore


def create_claims_agent_service(
    factory,
    *,
    similar_service=None,
    knowledge_retriever=None,
    similar_service_loader=None,
    knowledge_retriever_loader=None,
):
    """Construct once; retrieval components are loaded only when a tool needs them.

    Injected loader callbacks let FastAPI reuse its lifespan-scoped caches.
    The standalone script gets the same lazy behavior without constructing an app.
    """
    if factory is None:
        raise ValueError("Database is not configured")
    model = _required_model()
    similar_service_loader = _similar_loader(factory, similar_service, similar_service_loader)
    knowledge_retriever_loader = _knowledge_loader(knowledge_retriever, knowledge_retriever_loader)
    llm = _build_llm(model)
    tools = ReadOnlyAgentTools(
        factory,
        similar_service=similar_service,
        knowledge_retriever=knowledge_retriever,
        similar_service_loader=similar_service_loader,
        knowledge_retriever_loader=knowledge_retriever_loader,
    )
    return ClaimsAgentService(
        AgentServiceConfig(
            tools,
            StructuredAgentPlanner(llm),
            StructuredAgentFinalizer(llm),
            run_store=AgentRunStore(factory),
            llm_model=model,
        )
    )


def _required_model():
    model = os.environ.get("CLAIMS_LLM_MODEL")
    if not model:
        raise LlmError("CLAIMS_LLM_MODEL is required for agent generation")
    return model


def _similar_loader(factory, similar_service, similar_service_loader):
    if similar_service is not None or similar_service_loader is not None:
        return similar_service_loader
    from retrieval.similar_claims import create_similar_claims_service

    return lambda: create_similar_claims_service(factory)


def _knowledge_loader(knowledge_retriever, knowledge_retriever_loader):
    if knowledge_retriever is not None or knowledge_retriever_loader is not None:
        return knowledge_retriever_loader
    from knowledge.factory import create_knowledge_retriever

    return create_knowledge_retriever


def _build_llm(model):
    return OllamaStructuredLlm(
        model=model,
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        system_prompt=AGENT_SYSTEM_PROMPT,
        context_window=AGENT_LLM_CONTEXT_WINDOW,
    )
