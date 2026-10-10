"""Display semantic neighbors without LLM interpretation or invented metrics."""

import argparse
import json
from pathlib import Path

from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from retrieval.config import RetrievalConfig
from retrieval.representation import CLAIM_RETRIEVAL_REPRESENTATION_VERSION, build_claim_retrieval_text
from retrieval.similar_claims import SimilarClaimsService, create_retrieval_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", default="CLAIM-2026-00001")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional authoritative JSON response artifact")
    args = parser.parse_args()
    config = RetrievalConfig.from_env()
    engine = create_database_engine(DatabaseConfig.from_env())
    store = None
    try:
        repository, embeddings, store = create_retrieval_components(create_session_factory(engine), config)
        current = repository.get(args.claim_id)
        print(f"collection: {config.collection}\nembedding model: {config.embedding_model}\n"
              f"representation version: {CLAIM_RETRIEVAL_REPRESENTATION_VERSION}")
        print("Canonical query:\n" + build_claim_retrieval_text(current.facts))
        service = SimilarClaimsService(repository, embeddings, store, same_claim_type=config.same_claim_type)
        response = service.search(args.claim_id, args.limit)
        print(f"Top {args.limit} (similarity_score is not probability):")
        for result in response.results:
            print(f"{result.claim_id} | {result.incident_date} | {result.collision_type} | "
                  f"{result.repair_amount} {result.repair_currency} | similarity_score={result.similarity_score:.8f}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
    finally:
        if store is not None:
            store.close()
        engine.dispose()


if __name__ == "__main__":
    main()
