"""Cache."""

from pathlib import Path

from tests.integration_prerequisites._errors import PrerequisiteError


def require_cached_rag_model():
    from huggingface_hub import try_to_load_from_cache

    from retrieval.config import DEFAULT_EMBEDDING_MODEL

    def cached(filename):
        try:
            result = try_to_load_from_cache(DEFAULT_EMBEDDING_MODEL, filename)
        except OSError as error:
            raise PrerequisiteError(f"RAG embedding cache cannot be read: {error}") from None
        return isinstance(result, str) and Path(result).is_file()

    if not all(cached(name) for name in ("modules.json", "config.json")) or not any(
        cached(name) for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise PrerequisiteError(
            f"RAG embedding model is not present in the local cache: {DEFAULT_EMBEDDING_MODEL}.\n"
            "Populate its cache deliberately before running pytest -m rag. "
            "Pytest never downloads embedding models."
        )
