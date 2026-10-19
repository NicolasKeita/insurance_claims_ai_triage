"""Benchmark execution over a real retriever."""

from knowledge.evaluation.metrics import section_metrics
from knowledge.evaluation.models import RetrievalBenchmark
from knowledge.models import KnowledgeFilter


def evaluate_retrieval(retriever, benchmark: RetrievalBenchmark) -> dict:
    rows, answerable, unsupported = _collect_cases(retriever, benchmark.cases)
    metrics = {
        name: sum(r[name] for r in answerable) / len(answerable) if answerable else None
        for name in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")
    }
    return {
        "benchmark_version": benchmark.version,
        "evaluated_cases": len(rows),
        "answerable_cases": len(answerable),
        "unsupported_cases": len(unsupported),
        "ground_truth": "Exact source_id + section pairs; macro Recall@K, MRR truncated at 5; "
        "duplicate chunks count once for recall",
        **metrics,
        "unsupported": {
            "empty_result_rate": sum(r["returned_count"] == 0 for r in unsupported) / len(unsupported)
            if unsupported
            else None,
            "cases": unsupported,
            "interpretation": "Diagnostic only. Dense retrieval always ranks candidates; "
            "no calibrated abstention threshold. Unsupported cases excluded from recall/MRR; "
            "generation abstention tested separately.",
        },
        "cases": rows,
    }


def _collect_cases(retriever, cases):
    """Run each case and split metrics from the unsupported diagnostics."""
    rows, answerable, unsupported = [], [], []
    for case in cases:
        results = retriever.search(
            case.query,
            case.source_type,
            filters=KnowledgeFilter(product=case.product, language=case.language, as_of_date=case.as_of_date),
            limit=5,
        )
        row = {"case": case.model_dump(mode="json"), "results": [r.model_dump(mode="json") for r in results]}
        if case.answerable:
            row["metrics"] = section_metrics(results, case.expected_source_id, case.expected_sections)
            answerable.append(row["metrics"])
        else:
            row["unsupported"] = {
                "returned_count": len(results),
                "top_similarity_score": results[0].similarity_score if results else None,
            }
            unsupported.append(row["unsupported"])
        rows.append(row)
    return rows, answerable, unsupported
