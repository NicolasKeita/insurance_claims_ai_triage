"""Index synthetic Markdown sources; --recreate explicitly destroys only owned RAG collections."""

import argparse
import json
from pathlib import Path

from knowledge.config import ROOT
from knowledge.factory import create_knowledge_store
from knowledge.indexing import KnowledgeIndexer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="Delete/rebuild the two configured, owned RAG collections")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/rag/knowledge_index_manifest.json")
    args = parser.parse_args()
    store = create_knowledge_store()
    try:
        if args.recreate:
            print(f"Explicit destructive RAG rebuild: {store.config.policy_collection}, {store.config.procedure_collection}")
        print(json.dumps(KnowledgeIndexer(store).index_all(args.manifest, recreate=args.recreate), indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
