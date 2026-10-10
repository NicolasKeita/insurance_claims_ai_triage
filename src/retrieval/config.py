import os
from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COLLECTION = "insurance_claims_similar_v1"
DEFAULT_LIMIT = 5
MAX_LIMIT = 20


@dataclass(frozen=True)
class RetrievalConfig:
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    collection: str = DEFAULT_COLLECTION
    batch_size: int = 32
    same_claim_type: bool = True

    def __post_init__(self):
        if not self.collection or not self.embedding_model or not self.qdrant_url:
            raise ValueError("Retrieval URL, collection and model must be nonempty")
        if self.batch_size < 1 or self.batch_size > 1024:
            raise ValueError("Index batch size must be between 1 and 1024")

    @classmethod
    def from_env(cls):
        return cls(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            embedding_model=os.environ.get("CLAIMS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            collection=os.environ.get("CLAIMS_SIMILAR_COLLECTION", DEFAULT_COLLECTION),
            batch_size=int(os.environ.get("CLAIMS_INDEX_BATCH_SIZE", "32")),
        )


def validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIMIT}")
