"""Section-level retrieval metrics, separate from generation and abstention."""

import csv
import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowledge.models import KnowledgeSourceType


class RetrievalEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    query: str
    source_type: KnowledgeSourceType
    product: str | None
    language: str = "en"
    as_of_date: date | None = None
    expected_source_id: str | None
    expected_sections: list[str]
    answerable: bool

    @model_validator(mode="after")
    def truth(self):
        has_truth = bool(self.expected_source_id and self.expected_sections)
        if self.answerable != has_truth:
            raise ValueError("Answerable cases need source/sections; unsupported cases have no ground truth")
        if not self.answerable and (self.expected_source_id is not None or self.expected_sections):
            raise ValueError("Unsupported cases must have empty ground truth")
        return self


class RetrievalBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    cases: list[RetrievalEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Benchmark IDs must be unique")
        return self


def load_benchmark(path: Path) -> RetrievalBenchmark:
    return RetrievalBenchmark.model_validate_json(path.read_text(encoding="utf-8"))


def section_metrics(results, expected_source_id, expected_sections):
    truth = {(expected_source_id, s) for s in expected_sections}
    if not truth:
        raise ValueError("Recall/MRR require a nonempty answerable ground truth")
    values = {}
    for k in (1, 3, 5):
        found = {(r.source_id, r.section) for r in results[:k]}
        values[f"recall_at_{k}"] = len(found & truth) / len(truth)
    values["mrr"] = next((1 / rank for rank, r in enumerate(results[:5], 1)
                          if (r.source_id, r.section) in truth), 0.0)
    return values


def evaluate_retrieval(retriever, benchmark: RetrievalBenchmark) -> dict:
    rows, answerable, unsupported = [], [], []
    for case in benchmark.cases:
        results = retriever.search(case.query, case.source_type, product=case.product, language=case.language,
                                   as_of_date=case.as_of_date, limit=5)
        row = {"case": case.model_dump(mode="json"), "results": [r.model_dump(mode="json") for r in results]}
        if case.answerable:
            row["metrics"] = section_metrics(results, case.expected_source_id, case.expected_sections)
            answerable.append(row["metrics"])
        else:
            row["unsupported"] = {"returned_count": len(results),
                                  "top_similarity_score": results[0].similarity_score if results else None}
            unsupported.append(row["unsupported"])
        rows.append(row)
    metrics = {name: sum(r[name] for r in answerable) / len(answerable) if answerable else None
               for name in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")}
    return {
        "benchmark_version": benchmark.version, "evaluated_cases": len(rows),
        "answerable_cases": len(answerable), "unsupported_cases": len(unsupported),
        "ground_truth": "Exact source_id + section pairs; macro Recall@K, MRR truncated at 5; duplicate chunks count once for recall",
        **metrics,
        "unsupported": {
            "empty_result_rate": sum(r["returned_count"] == 0 for r in unsupported) / len(unsupported) if unsupported else None,
            "cases": unsupported,
            "interpretation": "Diagnostic only. Dense retrieval always ranks candidates; no calibrated abstention threshold. Unsupported cases excluded from recall/MRR; generation abstention tested separately.",
        },
        "cases": rows,
    }


def write_evaluation(report: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "retrieval_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "query", "answerable", "rank", "chunk_id", "source_id", "section", "similarity_score"])
        writer.writeheader()
        for row in report["cases"]:
            for rank, result in enumerate(row["results"], 1):
                writer.writerow({"id": row["case"]["id"], "query": row["case"]["query"],
                                 "answerable": row["case"]["answerable"], "rank": rank,
                                 **{k: result[k] for k in ("chunk_id", "source_id", "section", "similarity_score")}})
