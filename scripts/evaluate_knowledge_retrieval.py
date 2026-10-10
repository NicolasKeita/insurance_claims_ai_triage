"""Run retrieval benchmark with real embeddings and Qdrant; no LLM required."""

import argparse
import json
from pathlib import Path

from knowledge.config import ROOT
from knowledge.evaluation import evaluate_retrieval, load_benchmark, write_evaluation
from knowledge.factory import create_knowledge_retriever
from knowledge.models import KnowledgeSourceType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/evaluation/rag_retrieval_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rag")
    args = parser.parse_args()
    retriever = create_knowledge_retriever()
    try:
        report = evaluate_retrieval(retriever, load_benchmark(args.benchmark))
        report["collections"] = [retriever.store.contract(kind) | {"collection": retriever.store.config.collection(kind)}
                                 for kind in KnowledgeSourceType]
        write_evaluation(report, args.output_dir)
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
