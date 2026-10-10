import os
from dataclasses import dataclass
from pathlib import Path

from knowledge.models import KnowledgeSourceType
from retrieval.config import DEFAULT_EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_COLLECTION_VERSION = "knowledge_collections_v1"
KNOWLEDGE_CHUNKING_VERSION = "knowledge_chunking_v1"


@dataclass(frozen=True)
class KnowledgeConfig:
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    policy_collection: str = "insurance_policy_chunks_v1"
    procedure_collection: str = "insurance_procedure_chunks_v1"
    knowledge_dir: Path = ROOT / "data" / "knowledge"
    max_chunk_chars: int = 600
    batch_size: int = 32

    def __post_init__(self):
        if not all((self.qdrant_url, self.embedding_model, self.policy_collection, self.procedure_collection)):
            raise ValueError("RAG URL, model and collections must be nonempty")
        if self.policy_collection == self.procedure_collection:
            raise ValueError("POLICY and PROCEDURE must have separate collections")
        # Explicit --recreate may never target the existing claim index.
        claims_collection = os.environ.get("CLAIMS_SIMILAR_COLLECTION", "insurance_claims_similar_v1")
        if claims_collection in (self.policy_collection, self.procedure_collection):
            raise ValueError("RAG collections must differ from the Similar Claims collection")
        if not 1 <= self.batch_size <= 1024 or not 100 <= self.max_chunk_chars <= 10000:
            raise ValueError("Invalid RAG batch size or maximum chunk characters")

    def collection(self, source_type: KnowledgeSourceType) -> str:
        return self.policy_collection if KnowledgeSourceType(source_type) == KnowledgeSourceType.POLICY else self.procedure_collection

    @classmethod
    def from_env(cls):
        return cls(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            policy_collection=os.environ.get("RAG_POLICY_COLLECTION", "insurance_policy_chunks_v1"),
            procedure_collection=os.environ.get("RAG_PROCEDURE_COLLECTION", "insurance_procedure_chunks_v1"),
            max_chunk_chars=int(os.environ.get("RAG_MAX_CHUNK_CHARS", "600")),
            batch_size=int(os.environ.get("RAG_INDEX_BATCH_SIZE", "32")),
        )
