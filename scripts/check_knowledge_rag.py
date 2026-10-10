"""Separate real Ollama generation smoke test (synthetic knowledge only)."""

import argparse
import json
import logging
from pathlib import Path

from knowledge.config import ROOT
from knowledge.factory import create_knowledge_rag_service, create_knowledge_retriever
from knowledge.models import KnowledgeSourceType

SMOKE_CASES = [
    ("POLICY", "What is the AUTO_PREMIUM collision deductible and how is it applied?", "AUTO_PREMIUM", False),
    ("PROCEDURE", "What should the handler do when required supporting documents are missing?", None, False),
    ("POLICY", "Does AUTO_PREMIUM cover damage caused by volcanic ash?", "AUTO_PREMIUM", True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rag/generation_smoke.json")
    parser.add_argument("--debug", action="store_true", help="Explicit local query/debug logging; excerpts are printed by this demo script")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    retriever = create_knowledge_retriever()
    records = []
    try:
        service = create_knowledge_rag_service(retriever)
        for kind, question, product, expected_insufficient in SMOKE_CASES:
            chunks = retriever.search(question, KnowledgeSourceType(kind), product=product, language="en", limit=5)
            record = {"question": question, "source_type": kind,
                      "retrieved_chunks": [c.model_dump(mode="json") for c in chunks]}
            try:
                answer = service.answer_from_chunks(question, chunks)
                record["answer"] = answer.model_dump(mode="json")
                record["passed"] = answer.insufficient_evidence == expected_insufficient
            except Exception as error:
                record.update({"passed": False, "error": f"{type(error).__name__}: {error}"})
            records.append(record)
            print(json.dumps(record, indent=2, ensure_ascii=False))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"model": service.llm.model, "cases": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not all(r["passed"] for r in records):
            raise SystemExit("RAG smoke failed: inspect answers/citations and insufficient_evidence")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
