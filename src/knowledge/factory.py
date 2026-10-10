import os

from qdrant_client import QdrantClient

from claims.llm import LlmError, OllamaStructuredLlm
from knowledge.config import KnowledgeConfig
from knowledge.qdrant_store import QdrantKnowledgeStore
from knowledge.rag import KnowledgeRagService, RAG_SYSTEM_PROMPT
from knowledge.retrieval import KnowledgeRetriever
from retrieval.embeddings import SentenceTransformerEmbeddingProvider


def create_knowledge_store(config: KnowledgeConfig | None = None):
    config = config or KnowledgeConfig.from_env()
    embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model, asymmetric=True)
    client = QdrantClient(url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=10)
    return QdrantKnowledgeStore(client, embeddings, config)


def create_knowledge_retriever():
    return KnowledgeRetriever(create_knowledge_store())


def create_knowledge_rag_service(retriever):
    model = os.environ.get("CLAIMS_LLM_MODEL")
    if not model:
        raise LlmError("CLAIMS_LLM_MODEL is required for RAG generation")
    llm = OllamaStructuredLlm(model=model, host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                             system_prompt=RAG_SYSTEM_PROMPT)
    return KnowledgeRagService(retriever, llm)
