"""Index PostgreSQL facts. --recreate destroys ONLY the configured derived collection."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from persistence.database import DatabaseConfig, create_database_engine, create_session_factory
from retrieval.config import RetrievalConfig
from retrieval.similar_claims import SimilarClaimsIndexer, create_retrieval_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and rebuild the configured Qdrant collection; PostgreSQL is read only")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1]
                        / "artifacts" / "similar_claims" / "index_manifest.json")
    args = parser.parse_args()
    config = RetrievalConfig.from_env()
    engine = create_database_engine(DatabaseConfig.from_env())
    store = None
    try:
        repository, embeddings, store = create_retrieval_components(create_session_factory(engine), config)
        indexer = SimilarClaimsIndexer(repository, embeddings, store,
                                      config.batch_size if args.batch_size is None else args.batch_size)
        if args.recreate:
            print(f"Explicit rebuild: deleting derived Qdrant collection {config.collection}; PostgreSQL preserved")
        stats = indexer.index_all(recreate=args.recreate)
        manifest = indexer.write_manifest(args.manifest, stats)
        print(json.dumps({"stats": asdict(stats), "manifest": manifest}, indent=2))
    finally:
        if store is not None:
            store.close()
        engine.dispose()


if __name__ == "__main__":
    main()
